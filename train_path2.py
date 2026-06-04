"""
Path 2 training: multi-window joint SED with cross-window attention.

The architectural change moves cross-window reasoning from a post-hoc refinement
step (V152/V153 — δ near 0 because the per-window probabilities have already
collapsed information) to *before* the classifier head, where the 2048-dim
backbone features still carry information that joint reasoning can exploit.

Training data: V137 ensemble pseudo-labels on 10K soundscape files
                (HARD-thresholded, union'd with the 60 labeled soundscapes).
Loss: ASL on per-window hard labels.
Warm-start: EffNetV2-S R3 checkpoint (already converged on per-window task).
The cross-window attention block is initialized so cw_gate=0, giving the model
the option to ignore it entirely if joint reasoning doesn't help.

Usage:
  python train_path2.py --warm_ckpt checkpoints/effv2s_r3_fold0/best_fold0.pt \
                        --epochs 8 --batch_size 4 --experiment_name path2_v1
"""
import argparse
import json
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

from src.config import Config
from src.dataset_path2 import MultiWindowSoundscapeDataset
from src.models_path2 import MultiWindowSED
from src.losses import build_loss
from src.augmentations import TrainAugmentations
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--pseudo_csv", default="pseudo_labels_r4/raw_predictions.csv")
    p.add_argument("--soundscape_dir", default="data/train_soundscapes")
    p.add_argument("--soundscape_labels", default="data/train_soundscapes_labels.csv")
    p.add_argument("--warm_ckpt", default="checkpoints/effv2s_r3_fold0/best_fold0.pt")
    p.add_argument("--backbone", default="tf_efficientnetv2_s.in21k")
    p.add_argument("--n_mels", type=int, default=224)
    p.add_argument("--n_windows", type=int, default=12)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)         # files per step → 48 windows
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--drop_rate", type=float, default=0.2)
    p.add_argument("--cw_heads", type=int, default=4)
    p.add_argument("--cw_layers", type=int, default=2)
    p.add_argument("--cw_dropout", type=float, default=0.1)
    p.add_argument("--gamma_neg", type=float, default=4.0)
    p.add_argument("--clip_margin", type=float, default=0.05)
    p.add_argument("--soft_weight", type=float, default=0.0,
                   help="If >0, blend soft pseudo-label distillation into the loss.")
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--window_mask_prob", type=float, default=0.0,
                   help="Probability of applying window-mask augmentation per file. "
                        "Forces cross-window attention to engage to recover masked windows.")
    p.add_argument("--window_mask_max", type=int, default=3,
                   help="Max number of windows to mask when masking is applied.")
    p.add_argument("--freeze_gate_epochs", type=int, default=0,
                   help="Number of epochs at start to keep cw_gate fixed at its init "
                        "value. Lets cw_attn weights develop without the gate "
                        "suppressing them prematurely.")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment_name", default="path2_v1")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    return p.parse_args()


def make_config(args):
    c = Config()
    c.data_dir = args.data_dir
    c.backbone = args.backbone
    c.n_mels = args.n_mels
    c.batch_size = args.batch_size
    c.num_workers = args.num_workers
    c.lr = args.lr
    c.backbone_lr_mult = args.backbone_lr_mult
    c.weight_decay = args.weight_decay
    c.warmup_epochs = args.warmup_epochs
    c.epochs = args.epochs
    c.drop_rate = args.drop_rate
    c.gamma_pos = 0.0
    c.gamma_neg = args.gamma_neg
    c.clip_margin = args.clip_margin
    c.loss_type = "asl"
    c.amp = True
    c.max_grad_norm = args.max_grad_norm
    c.seed = args.seed
    c.experiment_name = args.experiment_name
    c.output_dir = args.output_dir
    c.mixup_alpha = 0.0
    c.mixup_prob = 0.0
    c.spec_augment = True
    c.freq_mask_param = 16
    c.time_mask_param = 24
    c.num_freq_masks = 2
    c.num_time_masks = 2
    return c


def build_scheduler(optimizer, config, steps_per_epoch):
    warmup = config.warmup_epochs * steps_per_epoch
    total = config.epochs * steps_per_epoch
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def soft_distill_loss(student_logits, teacher_probs, temperature):
    """Binary soft KD with sigmoid + temperature."""
    tau = temperature
    student_log_p = F.logsigmoid(student_logits / tau)
    student_log_1m = F.logsigmoid(-student_logits / tau)
    q = teacher_probs.clamp(1e-6, 1 - 1e-6)
    loss = q * (torch.log(q) - student_log_p) + (1 - q) * (torch.log(1 - q) - student_log_1m)
    return tau * tau * loss.mean()


@torch.no_grad()
def macro_auc_subset(preds, hard, mask):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(preds.shape[-1]):
        y = hard[..., c].flatten()
        p = preds[..., c].flatten()
        m = mask.flatten() > 0
        y, p = y[m], p[m]
        if y.sum() >= 3 and y.sum() < len(y):
            try:
                aucs.append(roc_auc_score(y, p))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


def main():
    args = parse_args()
    config = make_config(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(f"Device: {device}")
    print(f"Args: {json.dumps(vars(args), indent=2)}")

    # Labels
    tax = pd.read_csv(Path(args.data_dir) / "taxonomy.csv")
    tax["primary_label"] = tax["primary_label"].astype(str)
    label_cols = sorted(tax["primary_label"].tolist())
    config.num_classes = len(label_cols)

    augmentations = TrainAugmentations(config)
    ds = MultiWindowSoundscapeDataset(
        raw_predictions_csv=args.pseudo_csv,
        soundscape_dir=args.soundscape_dir,
        label_cols=label_cols,
        config=config,
        augmentations=augmentations,
        n_windows=args.n_windows,
        hard_label_csv_paths=[args.soundscape_labels],
        window_mask_prob=args.window_mask_prob,
        window_mask_max=args.window_mask_max,
    )

    n = len(ds)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(50, int(n * args.val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"Split: train={len(train_idx)} val={len(val_idx)}")

    val_ds_no_aug = MultiWindowSoundscapeDataset(
        raw_predictions_csv=args.pseudo_csv,
        soundscape_dir=args.soundscape_dir,
        label_cols=label_cols,
        config=config,
        augmentations=None,
        n_windows=args.n_windows,
        hard_label_csv_paths=[args.soundscape_labels],
    )
    train_loader = DataLoader(
        Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_ds_no_aug, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True,
    )

    # Model
    model = MultiWindowSED(
        config, n_windows=args.n_windows,
        cw_heads=args.cw_heads, cw_layers=args.cw_layers, cw_dropout=args.cw_dropout,
    ).to(device)

    # Warm-start from single-window checkpoint
    if args.warm_ckpt and Path(args.warm_ckpt).exists():
        ckpt = torch.load(args.warm_ckpt, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        loaded = model.load_single_window_state(sd)
        total = len(model.state_dict())
        print(f"Warm-start: loaded {loaded}/{total} tensors from {args.warm_ckpt}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {config.backbone} multi-window | {n_params:.1f}M params")

    optimizer = torch.optim.AdamW(model.get_param_groups(config), weight_decay=config.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)
    loss_fn = build_loss(config)
    scaler = GradScaler("cuda", enabled=config.amp)

    out_dir = Path(args.output_dir) / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    best_loss = 1e9
    best_auc = 0.0

    print(f"\n{'='*60}\nPath 2 training | {args.experiment_name}\n{'='*60}")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        gate_frozen = epoch < args.freeze_gate_epochs
        for step, (mels, soft, hard, mask) in enumerate(train_loader):
            mels = mels.to(device, non_blocking=True)
            hard = hard.to(device, non_blocking=True)
            B, W = mels.shape[:2]

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=config.amp):
                clip_logits, _ = model(mels)              # (B, W, C)
                # Per-window loss
                logits_flat = clip_logits.reshape(B * W, -1)
                hard_flat = hard.reshape(B * W, -1)
                loss = loss_fn(logits_flat, hard_flat)
                if args.soft_weight > 0:
                    soft_t = soft.to(device, non_blocking=True).reshape(B * W, -1)
                    loss = (1 - args.soft_weight) * loss + args.soft_weight * soft_distill_loss(
                        logits_flat, soft_t, args.temperature)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            # Freeze the cross-window gate during early epochs so cw_attn weights
            # can develop without the gate suppressing them prematurely.
            if gate_frozen and model.cw_gate.grad is not None:
                model.cw_gate.grad.zero_()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            losses.append(loss.item())
            if (step + 1) % 200 == 0:
                print(f"  ep{epoch+1} step{step+1}/{steps_per_epoch} loss={np.mean(losses[-200:]):.4f} "
                      f"gate={model.cw_gate.item():.4f} lr={scheduler.get_last_lr()[1]:.2e}", flush=True)

        train_loss = float(np.mean(losses))
        # Validate
        model.eval()
        val_preds, val_hard, val_mask = [], [], []
        with torch.no_grad():
            for mels, soft, hard, mask in val_loader:
                mels = mels.to(device, non_blocking=True)
                with autocast("cuda", enabled=config.amp):
                    clip_logits, _ = model(mels)
                val_preds.append(torch.sigmoid(clip_logits).float().cpu().numpy())
                val_hard.append(hard.numpy())
                val_mask.append(mask.numpy())
        val_preds = np.concatenate(val_preds)
        val_hard = np.concatenate(val_hard)
        val_mask = np.concatenate(val_mask)
        val_auc = macro_auc_subset(val_preds, val_hard, val_mask)

        elapsed = time.time() - t0
        gate = model.cw_gate.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss={train_loss:.4f} val_auc={val_auc:.4f} "
              f"gate={gate:.4f} {elapsed:.0f}s", flush=True)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(config),
                "label_cols": label_cols,
                "epoch": epoch + 1,
                "val_auc": val_auc,
                "args": vars(args),
            }, out_dir / "best.pt")
            print(f"  *BEST* val_auc={val_auc:.4f} saved")

        # Always save last
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": vars(config),
            "label_cols": label_cols,
            "epoch": epoch + 1,
            "val_auc": val_auc,
            "args": vars(args),
        }, out_dir / "last.pt")

    print(f"\nDone. best_val_auc={best_auc:.4f}, last gate={model.cw_gate.item():.4f}")


if __name__ == "__main__":
    main()
