"""
Finetune LB872 with SoftAUCLoss — directly optimize competition metric.

Strategy: Start from best Perch checkpoint (LB872), finetune with combined
ASL + SoftAUC loss. The SoftAUC component directly optimizes ranking quality
while ASL maintains good per-sample gradients.

Usage:
    python finetune_softauc.py --checkpoint kaggle_model/LB872.pt --epochs 5
"""
import os, sys, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.models_v2 import SEDModelV2
from src.dataset import BirdCLEFDataset, SoundscapeDataset
from src.losses import AsymmetricLoss, SoftAUCLoss, CombinedASLAUCLoss
from src.augmentations import TrainAugmentations, Mixup
from src.utils import set_seed, macro_auc, AverageMeter


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--asl_weight", type=float, default=0.5)
    parser.add_argument("--auc_weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--experiment_name", type=str, default="softauc_ft")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = Config(
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
        backbone_lr_mult=0.02,  # Very gentle backbone updates
        warmup_epochs=1, loss_type="asl",
    )

    # Data
    data_dir = Path("data")
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())
    num_classes = len(label_cols)

    df = pd.read_csv(data_dir / "train.csv")
    df["primary_label"] = df["primary_label"].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(df, df["primary_label"]))[args.fold]
    df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]

    augmentations = TrainAugmentations(config)
    train_ds = BirdCLEFDataset(df_train, data_dir / "train_audio", label_cols,
                                config, is_train=True, augmentations=augmentations)

    # Add labeled soundscapes
    sl_path = data_dir / "train_soundscapes_labels.csv"
    if sl_path.exists():
        sl_ds = SoundscapeDataset(sl_path, data_dir / "train_soundscapes",
                                   label_cols, config, is_train=True, augmentations=augmentations)
        train_ds = ConcatDataset([train_ds, sl_ds])
        print(f"Added {len(sl_ds)} soundscape windows")

    val_ds = BirdCLEFDataset(df_val, data_dir / "train_audio", label_cols,
                              config, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    # Load model from checkpoint
    model = SEDModelV2()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Optimizer: very gentle
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.gem_pool.parameters()) + list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * config.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = 1 * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        s / max(warmup_steps, 1) if s < warmup_steps else
        0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(total_steps - warmup_steps, 1))))

    # Loss: combined ASL + SoftAUC
    loss_fn = CombinedASLAUCLoss(
        asl_weight=args.asl_weight, auc_weight=args.auc_weight,
        temperature=args.temperature,
    )
    print(f"Loss: ASL({args.asl_weight}) + SoftAUC({args.auc_weight}) τ={args.temperature}")

    mixup = Mixup(alpha=0.3, prob=0.5)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path(f"checkpoints/{args.experiment_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline val AUC before finetuning
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, labels in val_loader:
            mel = mel.to(device)
            probs = model(mel)["clipwise_prob"].cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.numpy())
    baseline_auc = macro_auc(np.concatenate(all_labels), np.concatenate(all_preds))
    print(f"\nBaseline val_auc: {baseline_auc:.4f}")
    best_auc = baseline_auc

    print(f"\n{'='*60}")
    print(f"SoftAUC Finetuning | {args.epochs} epochs | lr={args.lr}")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        model.train()
        meter = AverageMeter()
        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)
            mel, labels = mixup(mel, labels)
            with autocast("cuda", enabled=device.type == "cuda"):
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

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mel, labels in val_loader:
                mel = mel.to(device)
                probs = model(mel)["clipwise_prob"].cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.numpy())
        val_auc = macro_auc(np.concatenate(all_labels), np.concatenate(all_preds))

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1, "val_auc": val_auc,
                "label_cols": label_cols,
            }, out_dir / f"best_fold{args.fold}.pt")

        lr_now = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss={meter.avg:.4f} | "
              f"val_auc={val_auc:.4f} | lr={lr_now:.2e}"
              f"{' *BEST*' if is_best else ''}")

    print(f"\nBaseline: {baseline_auc:.4f} → Best: {best_auc:.4f} "
          f"({'improved' if best_auc > baseline_auc else 'no improvement'})")


if __name__ == "__main__":
    main()
