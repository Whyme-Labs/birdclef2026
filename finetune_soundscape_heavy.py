"""V219 soundscape-heavy finetune: upsample 1478 labeled soundscapes 30x
within the focal+pseudo training mix, so they contribute ~25% of training signal
instead of the default 1.2%.

Site-stratified file-level val split (16 files held out across 9 sites).

Warm-start V137 V2S R2_fold1 (LB 0.942 single-model). Soup result with R2_fold1
to recover focal calibration on the 159 classes that don't appear in soundscapes.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
import sys, os

sys.path.insert(0, ".")
from src.config import Config
from src.dataset import BirdCLEFDataset, SoundscapeDataset, PseudoLabelDataset
from src.augmentations import TrainAugmentations
from src.models_v2 import SEDModelV2
from src.losses import AsymmetricLoss
from src.utils import set_seed


def site_stratified_val_files(labels_csv: str, val_fraction: float = 0.25, seed: int = 42):
    """Split files by site, hold out ~val_fraction per site."""
    df = pd.read_csv(labels_csv)
    df["site"] = df["filename"].str.extract(r"_S(\d+)_")[0]
    rng = np.random.default_rng(seed)
    val_files = []
    for site, group in df.groupby("site"):
        files = group["filename"].unique()
        n_val = max(1, int(round(len(files) * val_fraction)))
        chosen = rng.choice(files, size=n_val, replace=False)
        val_files.extend(chosen.tolist())
    return set(val_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/effv2s_r2_fold1/best_fold1.pt")
    ap.add_argument("--out_dir", default="checkpoints/v219_soundscape_heavy")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--soundscape_repeat", type=int, default=30,
                    help="Repeat factor for labeled soundscape windows")
    ap.add_argument("--use_pseudo", action="store_true",
                    help="Also include V137 pseudo on focal recordings")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = Config(backbone="tf_efficientnetv2_s.in21k")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Labels
    tax = pd.read_csv("data/taxonomy.csv")
    label_cols = sorted(tax["primary_label"].astype(str).tolist())

    # Site-stratified val split
    sl_path = "data/train_soundscapes_labels.csv"
    val_files = site_stratified_val_files(sl_path)
    print(f"Held out {len(val_files)} files for val")

    # Build full soundscape dataset, partition by val_files
    full_sl = SoundscapeDataset(sl_path, "data/train_soundscapes",
                                label_cols, cfg, is_train=True,
                                augmentations=TrainAugmentations(cfg))
    train_idx = [i for i, fn in enumerate(full_sl.df["filename"]) if fn not in val_files]
    val_idx = [i for i, fn in enumerate(full_sl.df["filename"]) if fn in val_files]
    print(f"Soundscape train: {len(train_idx)} | val: {len(val_idx)}")

    sl_train = Subset(full_sl, train_idx)
    sl_val_ds = SoundscapeDataset(sl_path, "data/train_soundscapes",
                                  label_cols, cfg, is_train=False,
                                  augmentations=None)
    sl_val_idx = [i for i, fn in enumerate(sl_val_ds.df["filename"])
                  if fn in val_files]
    sl_val = Subset(sl_val_ds, sl_val_idx)

    # Repeat soundscape train N×
    train_parts = [sl_train] * args.soundscape_repeat
    print(f"Soundscape repeated {args.soundscape_repeat}x → "
          f"{len(sl_train) * args.soundscape_repeat} effective rows")

    # Add focal recordings (full set, no pseudo for now to keep it clean)
    train_df = pd.read_csv("data/train.csv")
    focal_ds = BirdCLEFDataset(train_df, "data/train_audio", label_cols, cfg,
                               is_train=True,
                               augmentations=TrainAugmentations(cfg))
    train_parts.append(focal_ds)

    if args.use_pseudo:
        pl_csv = "pseudo_labels_v137/raw_predictions.csv"
        pl_ds = PseudoLabelDataset(pl_csv, "data/train_soundscapes",
                                   label_cols, cfg, is_train=True,
                                   augmentations=TrainAugmentations(cfg))
        train_parts.append(pl_ds)
        print(f"Added {len(pl_ds)} pseudo windows")

    train_set = ConcatDataset(train_parts)
    print(f"Total train samples: {len(train_set)} ({len(focal_ds)} focal + "
          f"{args.soundscape_repeat}×{len(sl_train)} soundscape)")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(sl_val, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model: warm-start from V137's V2S R2 fold 1
    model = SEDModelV2(backbone=cfg.backbone, num_classes=len(label_cols)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  load: missing={len(missing)} unexpected={len(unexpected)}")

    loss_fn = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip_margin=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=args.lr * 0.1
    )
    scaler = torch.amp.GradScaler("cuda")

    best_val = 0.0
    print(f"\n{'='*60}\nSoundscape-heavy FT | {args.epochs} epochs | lr={args.lr}")
    print(f"Train: {len(train_set)} | Val (held-out soundscape): {len(sl_val)}")
    print(f"{'='*60}\n")

    from sklearn.metrics import roc_auc_score
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for step, (mel, lbl) in enumerate(train_loader):
            mel = mel.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(mel)["clipwise_logit"]
                loss = loss_fn(logits, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            loss_sum += loss.item(); n += 1
            if step % 100 == 0:
                print(f"  ep {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss_sum/n:.4f}", flush=True)

        # Val on held-out soundscapes
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for mel, lbl in val_loader:
                mel = mel.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(mel)["clipwise_logit"]
                p = torch.sigmoid(logits).float().cpu().numpy()
                all_p.append(p); all_y.append(lbl.numpy())
        all_p = np.concatenate(all_p); all_y = np.concatenate(all_y)
        valid = (all_y > 0.5).sum(0) > 0
        try:
            val_auc = roc_auc_score(all_y[:, valid], all_p[:, valid], average="macro")
        except Exception:
            val_auc = 0.0

        is_best = val_auc > best_val
        marker = " *BEST*" if is_best else ""
        print(f"Epoch {epoch}/{args.epochs} | loss={loss_sum/n:.4f} | "
              f"sl_val_auc={val_auc:.4f}{marker}", flush=True)

        ckpt_out = {"model_state_dict": model.state_dict(), "val_auc": val_auc, "epoch": epoch}
        torch.save(ckpt_out, out_dir / f"epoch_{epoch}.pt")
        if is_best:
            best_val = val_auc
            torch.save(ckpt_out, out_dir / "best.pt")

    print(f"\nBest sl_val_auc: {best_val:.4f}")


if __name__ == "__main__":
    main()
