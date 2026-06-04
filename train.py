"""
BirdCLEF+ 2026 Training Script.

Training strategy:
──────────────────
1. Differential learning rate: backbone at 0.1× (preserve pretrained features),
   head at 1× (train from scratch). Supported by Kornblith et al. (CVPR 2019).

2. Cosine annealing with linear warmup: warmup prevents large early gradients
   from destroying pretrained backbone features. Cosine decay smoothly reduces
   LR, empirically better than step decay for fine-tuning (Loshchilov & Hutter,
   ICLR 2017).

3. Mixed precision (AMP): ~2× throughput on tensor cores with negligible
   accuracy impact. Critical for fitting larger batches on 11GB GPU.

4. Class-balanced sampling via inverse-sqrt weighting: addresses the extreme
   long-tail distribution (1 to 499 recordings per species).

5. Gradient accumulation: effective batch size = batch_size × grad_accum_steps.
   Larger effective batches stabilize multi-label training.

Usage:
  python train.py --backbone tf_efficientnet_b0.ns_jft_in1k --fold 0
  python train.py --backbone tf_efficientnet_b3.ns_jft_in1k --fold 0 --batch_size 16 --experiment_name teacher_b3
"""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path

from src.config import config_from_args
from src.dataset import BirdCLEFDataset, SoundscapeDataset, PseudoLabelDataset, build_splits, build_sampler
from src.models import SEDModel
from src.losses import build_loss
from src.augmentations import TrainAugmentations, Mixup
from src.utils import set_seed, macro_auc, AverageMeter


def build_scheduler(optimizer, config, steps_per_epoch):
    """
    Cosine annealing with linear warmup.

    LR schedule:
      t < warmup: lr(t) = lr_max × (t / warmup_steps)
      t ≥ warmup: lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × (t-warmup)/(total-warmup)))

    This is the standard schedule for fine-tuning pretrained models.
    """
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, mixup,
                    scaler, config, device):
    model.train()
    meter = AverageMeter()
    total_steps = len(loader)

    optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
    for step, (mel, labels) in enumerate(loader):
        mel = mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Batch-level mixup
        if mixup is not None:
            mel, labels = mixup(mel, labels)

        # Forward with mixed precision
        with autocast("cuda", enabled=config.amp):
            clip_logits, _ = model(mel)
            loss = loss_fn(clip_logits, labels)
            loss = loss / config.grad_accum_steps

        # Backward
        scaler.scale(loss).backward()

        if (step + 1) % config.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        meter.update(loss.item() * config.grad_accum_steps, mel.size(0))

        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{total_steps} loss={meter.avg:.4f}", flush=True)

    return meter.avg


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for mel, labels in loader:
        mel = mel.to(device, non_blocking=True)
        clip_logits, _ = model(mel)
        probs = torch.sigmoid(clip_logits).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auc = macro_auc(all_labels, all_preds)
    return auc


def main():
    config = config_from_args()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Performance optimizations ──
    torch.backends.cudnn.benchmark = True  # Auto-tune conv algorithms for fixed input size
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matmul (2x throughput on Ampere+)
    torch.backends.cudnn.allow_tf32 = True  # TF32 for cuDNN
    torch.set_float32_matmul_precision("high")  # Enable TF32 globally

    print(f"Device: {device}")
    print(f"Config: {json.dumps({k: v for k, v in vars(config).items()}, indent=2)}")

    # Data
    data_dir = Path(config.data_dir)
    train_csv_path = Path(config.train_csv)
    if not train_csv_path.is_absolute():
        train_csv_path = data_dir / train_csv_path
    df_train, df_val, label_cols = build_splits(
        train_csv_path, data_dir / "taxonomy.csv", config
    )
    print(f"Train CSV: {train_csv_path}")
    print(f"Train: {len(df_train)} samples, Val: {len(df_val)} samples")
    print(f"Species: {len(label_cols)} total, "
          f"{df_train['primary_label'].nunique()} in train, "
          f"{df_val['primary_label'].nunique()} in val")
    if "source" in df_train.columns:
        print("Train source counts:")
        print(df_train.groupby("source").size().to_string())

    # Resolve external audio dir (absolute or relative to project root).
    ext_2025_path = Path(config.external_2025_audio_dir)
    if not ext_2025_path.is_absolute():
        ext_2025_path = Path.cwd() / ext_2025_path
    extra_audio_dirs = {
        "2026": data_dir / "train_audio",
        "2025": ext_2025_path,
        "XC": Path("data_external/xc_pantanal_audio"),
    }

    # Datasets
    augmentations = TrainAugmentations(config)
    train_ds = BirdCLEFDataset(
        df_train, data_dir / "train_audio", label_cols, config,
        is_train=True, augmentations=augmentations,
        extra_audio_dirs=extra_audio_dirs,
    )

    # Add labeled soundscape data — this covers the 28 zero-shot species
    soundscape_labels = data_dir / "train_soundscapes_labels.csv"
    if soundscape_labels.exists():
        soundscape_ds = SoundscapeDataset(
            soundscape_labels, data_dir / "train_soundscapes",
            label_cols, config, is_train=True, augmentations=augmentations
        )
        train_ds = torch.utils.data.ConcatDataset([train_ds, soundscape_ds])
        print(f"Added {len(soundscape_ds)} soundscape windows to training")

    # Add pseudo-labeled soundscape data — the main domain adaptation lever
    pseudo_dir = Path(config.pseudo_label_dir)
    if config.pseudo_label_dir and (pseudo_dir / "raw_predictions.csv").exists():
        pseudo_ds = PseudoLabelDataset(
            pseudo_dir / "raw_predictions.csv",
            data_dir / "train_soundscapes",
            label_cols, config,
            augmentations=augmentations,
            label_weight=config.pseudo_label_weight,
        )
        train_ds = torch.utils.data.ConcatDataset([train_ds, pseudo_ds])
        print(f"Added {len(pseudo_ds)} pseudo-labeled windows "
              f"(weight={config.pseudo_label_weight})")

    val_ds = BirdCLEFDataset(
        df_val, data_dir / "train_audio", label_cols, config,
        is_train=False, augmentations=None,
        extra_audio_dirs=extra_audio_dirs,
    )

    # Multi-source weighted sampler for class-balanced training
    from src.dataset import build_concat_sampler
    sc_len = len(soundscape_ds) if soundscape_labels.exists() else 0
    ps_len = len(pseudo_ds) if (config.pseudo_label_dir and
              (pseudo_dir / "raw_predictions.csv").exists()) else 0
    sampler = build_concat_sampler(
        df_train, label_cols,
        soundscape_len=sc_len, pseudo_len=ps_len,
        power=0.5, soundscape_weight=3.0, pseudo_weight=0.5,
    )
    print(f"Sampler: sqrt-balanced, {len(df_train)} focal + {sc_len} soundscape(3x) + {ps_len} pseudo(0.5x)")
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True,
    )

    # Model
    model = SEDModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model: {config.backbone}, {total_params:.1f}M params ({trainable_params:.1f}M trainable)")

    if config.resume_from:
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"Resumed model weights from {config.resume_from} (epoch {ckpt.get('epoch')}, val_auc {ckpt.get('val_auc')})")

    # torch.compile disabled — GTX 1080 Ti (CC 6.1) not supported by Triton
    # try:
    #     model = torch.compile(model, mode="reduce-overhead")
    #     print("torch.compile enabled (reduce-overhead mode)")
    # except Exception as e:
    #     print(f"torch.compile skipped: {e}")
    print("torch.compile disabled (GPU CC < 7.0)")

    # Optimizer with differential LR
    param_groups = model.get_param_groups(config)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    steps_per_epoch = len(train_loader) // config.grad_accum_steps
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)

    # Loss
    loss_fn = build_loss(config)
    print(f"Loss: {config.loss_type}")

    # Mixup
    mixup = Mixup(alpha=config.mixup_alpha, prob=config.mixup_prob)

    # AMP scaler
    scaler = GradScaler("cuda", enabled=config.amp)

    # Output directory
    out_dir = Path(config.output_dir) / config.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    # Training loop
    best_auc = 0.0
    best_epoch = 0
    print(f"\n{'='*60}")
    print(f"Training {config.experiment_name} | Fold {config.fold}")
    print(f"{'='*60}")

    for epoch in range(config.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            mixup, scaler, config, device
        )
        val_auc = validate(model, val_loader, device)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[1]["lr"]  # Head LR
        print(f"Epoch {epoch+1:02d}/{config.epochs} | "
              f"loss={train_loss:.4f} | val_auc={val_auc:.4f} | "
              f"lr={lr_now:.2e} | {elapsed:.0f}s", end="")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": vars(config),
                "epoch": epoch + 1,
                "val_auc": val_auc,
                "label_cols": label_cols,
            }
            torch.save(ckpt, out_dir / f"best_fold{config.fold}.pt")
            print(" *BEST*")
        else:
            print()

    print(f"\nBest: epoch {best_epoch}, val_auc={best_auc:.4f}")
    print(f"Saved to {out_dir / f'best_fold{config.fold}.pt'}")


if __name__ == "__main__":
    main()
