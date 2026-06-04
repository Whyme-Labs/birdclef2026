"""
Fine-tune the public LB872 checkpoint on our data.

Strategy: start from Perch-finetuned weights (LB872), fine-tune with very low LR
on labeled recordings + labeled soundscapes. Short training (10 epochs) since the
model is already well-trained — we're adapting, not training from scratch.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.models_v2 import SEDModelV2
from src.dataset import BirdCLEFDataset, SoundscapeDataset, PseudoLabelDataset
from src.losses import AsymmetricLoss
from src.augmentations import TrainAugmentations, Mixup
from src.utils import set_seed, macro_auc, AverageMeter


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="finetune_lb872")
    parser.add_argument("--pseudo_label_dir", type=str, default=None,
                        help="Directory containing raw_predictions.csv from pseudo-labeling")
    parser.add_argument("--loss_type", type=str, default="asl",
                        choices=["asl", "combined_asl_auc"],
                        help="Loss function: asl (default) or combined_asl_auc")
    parser.add_argument("--soundscape_only", action="store_true",
                        help="Train only on soundscape data (no focal recordings). "
                             "Preserves soundscape calibration of Perch models.")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
        backbone="tf_efficientnetv2_s.in21k",  # match V137 V2S R2 checkpoint
        n_mels=224, fmin=0.0, fmax=16000.0,
        backbone_lr_mult=0.05,  # Very gentle backbone updates
        warmup_epochs=1, loss_type=args.loss_type,
    )

    # Data
    data_dir = Path("data")
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())

    df = pd.read_csv(data_dir / "train.csv")
    df["primary_label"] = df["primary_label"].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(df, df["primary_label"]))[args.fold]
    df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]

    augmentations = TrainAugmentations(config)

    if args.soundscape_only:
        # Soundscape-only mode: skip focal recordings to preserve soundscape calibration
        # Exp 6/7 showed fine-tuning on recordings degrades LB (0.890 → 0.882)
        train_datasets = []
        print("Mode: soundscape-only (no focal recordings)")
    else:
        train_ds = BirdCLEFDataset(df_train, data_dir / "train_audio", label_cols, config,
                                    is_train=True, augmentations=augmentations)
        train_datasets = [train_ds]

    # Add labeled soundscapes
    sl_path = data_dir / "train_soundscapes_labels.csv"
    if sl_path.exists():
        sl_ds = SoundscapeDataset(sl_path, data_dir / "train_soundscapes",
                                   label_cols, config, is_train=True, augmentations=augmentations)
        train_datasets.append(sl_ds)
        print(f"Added {len(sl_ds)} soundscape windows")

    # Add pseudo-labeled soundscapes
    if args.pseudo_label_dir:
        pl_csv = Path(args.pseudo_label_dir) / "raw_predictions.csv"
        if pl_csv.exists():
            pl_ds = PseudoLabelDataset(pl_csv, data_dir / "train_soundscapes",
                                        label_cols, config, augmentations=augmentations,
                                        label_weight=0.5)
            train_datasets.append(pl_ds)
            print(f"Added {len(pl_ds)} pseudo-labeled windows")

    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    val_ds = BirdCLEFDataset(df_val, data_dir / "train_audio", label_cols, config,
                              is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    # Soundscape validation: evaluate on labeled soundscapes for model selection
    # In soundscape_only mode, focal val_auc declines as model adapts to soundscapes,
    # so we need a soundscape-based metric for early stopping.
    sl_eval_loader = None
    if sl_path.exists():
        sl_eval_ds = SoundscapeDataset(sl_path, data_dir / "train_soundscapes",
                                        label_cols, config, is_train=False)
        sl_eval_loader = DataLoader(sl_eval_ds, batch_size=args.batch_size * 2,
                                     shuffle=False, num_workers=2, pin_memory=True)
        print(f"Soundscape validation: {len(sl_eval_ds)} windows")

    # Load pretrained model — use config backbone, not the default B0.
    # strict=False because R2 fold 0 checkpoint uses the older head/gem naming
    # (gem.p, head.attention.weight) — backbone weights still load; head reinits.
    model = SEDModelV2(backbone=config.backbone, num_classes=len(label_cols))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    missing[:5]: {missing[:5]}")
    if unexpected:
        print(f"    unexpected[:5]: {unexpected[:5]}")
    model.to(device)
    print(f"Loaded {args.checkpoint}")
    print(f"  Original metrics: {ckpt.get('metrics', {})}")

    # Optimizer: differential LR
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.gem_pool.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * config.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-5)

    total_steps = args.epochs * (len(train_loader) // 1)
    warmup_steps = 1 * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        s / max(warmup_steps, 1) if s < warmup_steps else
        0.5 * (1 + np.cos(np.pi * (s - warmup_steps) / max(total_steps - warmup_steps, 1))))

    if args.loss_type == "combined_asl_auc":
        from src.losses import CombinedASLAUCLoss
        loss_fn = CombinedASLAUCLoss(asl_weight=0.7, auc_weight=0.3)
        print(f"Loss: CombinedASLAUCLoss (0.7 ASL + 0.3 SoftAUC)")
    else:
        loss_fn = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip_margin=0.05)
    mixup = Mixup(alpha=0.5, prob=0.5)
    scaler = GradScaler("cuda", enabled=True)

    out_dir = Path(f"checkpoints/{args.experiment_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    print(f"\n{'='*60}")
    print(f"Fine-tuning {args.experiment_name} | {args.epochs} epochs | lr={args.lr}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        # Train
        model.train()
        meter = AverageMeter()
        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)
            mel, labels = mixup(mel, labels)
            with autocast("cuda"):
                out = model(mel)
                loss = loss_fn(out["clipwise_logit"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            meter.update(loss.item(), mel.size(0))

        # Validate (focal recordings)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mel, labels in val_loader:
                mel = mel.to(device)
                probs = model(mel)["clipwise_prob"].cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.numpy())
        val_auc = macro_auc(np.concatenate(all_labels), np.concatenate(all_preds))

        # Soundscape validation (labeled soundscapes)
        sl_auc = None
        if sl_eval_loader is not None:
            sl_preds, sl_labels = [], []
            with torch.no_grad():
                for mel, labels in sl_eval_loader:
                    mel = mel.to(device)
                    probs = model(mel)["clipwise_prob"].cpu().numpy()
                    sl_preds.append(probs)
                    sl_labels.append(labels.numpy())
            sl_auc = macro_auc(np.concatenate(sl_labels), np.concatenate(sl_preds))

        # Model selection: use soundscape AUC in soundscape_only mode, focal otherwise
        select_auc = sl_auc if (args.soundscape_only and sl_auc is not None) else val_auc
        lr_now = optimizer.param_groups[1]["lr"]
        is_best = select_auc > best_auc
        if is_best:
            best_auc = select_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1, "val_auc": val_auc,
                "sl_auc": sl_auc,
                "label_cols": label_cols,
            }, out_dir / f"best_fold{args.fold}.pt")

        # Also save every epoch for offline evaluation
        if args.soundscape_only:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1, "val_auc": val_auc,
                "sl_auc": sl_auc,
                "label_cols": label_cols,
            }, out_dir / f"epoch{epoch+1}_fold{args.fold}.pt")

        sl_str = f" | sl_auc={sl_auc:.4f}" if sl_auc is not None else ""
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss={meter.avg:.4f} | "
              f"val_auc={val_auc:.4f}{sl_str} | lr={lr_now:.2e}"
              f"{' *BEST*' if is_best else ''}")

    metric_name = "sl_auc" if args.soundscape_only else "val_auc"
    print(f"\nBest: {metric_name}={best_auc:.4f}")


if __name__ == "__main__":
    main()
