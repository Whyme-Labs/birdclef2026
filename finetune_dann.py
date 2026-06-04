"""
DANN-style domain-adversarial training for V173.

Closes focal→soundscape distribution gap explicitly:
  - Backbone: EffNetV2-S in21k SED (same as R2/R5)
  - Species head: 234-class (the main task)
  - Domain head: 2-class (focal vs soundscape) preceded by Gradient Reversal Layer
  - Loss: species_ASL + λ * domain_CE

When backbone minimizes species loss but MAXIMIZES domain loss (via GRL),
learned features become DOMAIN-INVARIANT — the same regardless of whether
input is focal or soundscape. This addresses the V168/V170 failure mode where
strong-aug models had high sl_auc but failed on LB (focal/soundscape gap).

Reference: Ganin et al. "Unsupervised Domain Adaptation by Backpropagation" (ICML 2015)
Bioacoustic application: arXiv 2507.13727 reports +10.5% relative on bioacoustic shift.

Usage:
  python finetune_dann.py --epochs 5 --lr 5e-4 --batch_size 40 \\
      --pseudo_label_dir pseudo_labels_r4 --experiment_name effv2s_dann
"""
import os, time, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.models import SEDModel
from src.dataset import BirdCLEFDataset, SoundscapeDataset, PseudoLabelDataset
from src.losses import AsymmetricLoss
from src.utils import set_seed, macro_auc, AverageMeter


# ──────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer (DANN core primitive)
# ──────────────────────────────────────────────────────────────────────────────
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return GradientReversalFunction.apply(x, lambda_)


class DomainHead(nn.Module):
    """2-layer MLP for binary domain classification (focal vs soundscape)."""
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class DANN(nn.Module):
    """SEDModel + domain head with Gradient Reversal Layer."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sed = SEDModel(config)
        # Domain head takes pooled backbone features (B, feat_dim) — pool over time after GEM
        self.domain_head = DomainHead(self.sed.feat_dim, hidden=128)

    def forward(self, x, lambda_=0.0):
        # Backbone → (B, C, F', T'); after gem → (B, C, T'); transpose → (B, T', C)
        features = self.sed.backbone(x)            # (B, C, F', T')
        features = self.sed.gem(features)           # (B, C, T')
        # For species head: SED-style attention pooling
        feat_t = features.permute(0, 2, 1)          # (B, T', C)
        feat_dropped = self.sed.dropout(feat_t)
        clip_logits, frame_logits = self.sed.head(feat_dropped)
        # For domain head: pool over time, then GRL
        pooled = features.mean(dim=2)               # (B, C)
        pooled_grl = grad_reverse(pooled, lambda_)
        domain_logits = self.domain_head(pooled_grl)  # (B, 2)
        return clip_logits, frame_logits, domain_logits


# ──────────────────────────────────────────────────────────────────────────────
# Dataset wrappers that emit (mel, species_label, domain_label)
# ──────────────────────────────────────────────────────────────────────────────
class DomainTaggedDataset(Dataset):
    """Wraps a base dataset to add a fixed domain label per sample."""
    def __init__(self, base, domain_label):
        self.base = base
        self.domain_label = int(domain_label)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]
        # base may return (mel, species_label) or (mel, species_label, weight)
        if len(out) == 2:
            mel, lbl = out
        else:
            mel, lbl = out[0], out[1]
        return mel, lbl, torch.tensor(self.domain_label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Strong augmentation in batch (mixup + specaug only — no soundscape noise mix)
# ──────────────────────────────────────────────────────────────────────────────
import torchaudio


class StrongSpecAugment(nn.Module):
    def __init__(self, freq_mask_param=40, time_mask_param=60,
                 num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, mel):
        for _ in range(self.num_freq_masks):
            mel = self.freq_mask(mel)
        for _ in range(self.num_time_masks):
            mel = self.time_mask(mel)
        return mel


class StrongMixup:
    def __init__(self, alpha=1.0, prob=0.8):
        self.alpha = alpha; self.prob = prob

    def __call__(self, mel, labels, domain):
        if torch.rand(1).item() > self.prob:
            return mel, labels, domain
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(mel.size(0), device=mel.device)
        mixed_mel = lam * mel + (1 - lam) * mel[idx]
        mixed_labels = lam * labels + (1 - lam) * labels[idx]
        # For domain: keep majority (lam>0.5 keeps original; otherwise the mixed sample's domain)
        # Simple choice: keep original domain (we mix species labels but not domain identity)
        return mixed_mel, mixed_labels, domain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo_label_dir", default="pseudo_labels_r4")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=40)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--experiment_name", default="effv2s_dann")
    ap.add_argument("--lambda_max", type=float, default=0.1,
                    help="DANN λ ramps 0→lambda_max over training")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(
        backbone="tf_efficientnetv2_s.in21k",
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
        backbone_lr_mult=0.1, warmup_epochs=2,
        loss_type="combined_asl_auc",
        n_mels=224, fmin=0.0, fmax=16000.0,
        in_chans=3, gem_p_init=3.0, drop_rate=0.3,
        weight_decay=1e-5, max_grad_norm=1.0,
    )
    config.num_classes = 234

    data_dir = Path("data")
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())

    df = pd.read_csv(data_dir / "train.csv")
    df["primary_label"] = df["primary_label"].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(df, df["primary_label"]))[args.fold]
    df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]

    # Datasets — focal recordings get domain=0, soundscape sources get domain=1
    train_focal_base = BirdCLEFDataset(df_train, data_dir / "train_audio",
                                        label_cols, config, is_train=True, augmentations=None)
    train_focal = DomainTaggedDataset(train_focal_base, domain_label=0)

    sl_path = data_dir / "train_soundscapes_labels.csv"
    sl_base = SoundscapeDataset(sl_path, data_dir / "train_soundscapes",
                                 label_cols, config, is_train=True, augmentations=None)
    sl_ds = DomainTaggedDataset(sl_base, domain_label=1)
    print(f"Focal: {len(train_focal)}, Soundscape: {len(sl_ds)}")

    pl_csv = Path(args.pseudo_label_dir) / "raw_predictions.csv"
    pl_ds = None
    if pl_csv.exists():
        pl_base = PseudoLabelDataset(pl_csv, data_dir / "train_soundscapes",
                                      label_cols, config, augmentations=None,
                                      label_weight=0.5)
        pl_ds = DomainTaggedDataset(pl_base, domain_label=1)
        print(f"Pseudo-labeled R4: {len(pl_ds)} (treated as soundscape domain)")

    train_ds = ConcatDataset([train_focal, sl_ds] + ([pl_ds] if pl_ds else []))
    val_ds = BirdCLEFDataset(df_val, data_dir / "train_audio", label_cols, config,
                              is_train=False)
    sl_eval_ds = SoundscapeDataset(sl_path, data_dir / "train_soundscapes",
                                    label_cols, config, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    sl_loader = DataLoader(sl_eval_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model
    model = DANN(config).to(device)
    print(f"DANN model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params "
          f"(species_head + domain_head + GRL)")

    backbone_params = list(model.sed.backbone.parameters())
    species_params = list(model.sed.gem.parameters()) + list(model.sed.head.parameters())
    domain_params = list(model.domain_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * config.backbone_lr_mult},
        {"params": species_params, "lr": args.lr},
        {"params": domain_params, "lr": args.lr},
    ], weight_decay=1e-5)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = config.warmup_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        s / max(warmup_steps, 1) if s < warmup_steps else
        0.5 * (1 + np.cos(np.pi * (s - warmup_steps) / max(total_steps - warmup_steps, 1))))

    from src.losses import CombinedASLAUCLoss
    species_loss_fn = CombinedASLAUCLoss(asl_weight=0.7, auc_weight=0.3)
    domain_loss_fn = nn.CrossEntropyLoss()
    spec_aug = StrongSpecAugment().to(device)
    mixup = StrongMixup(alpha=1.0, prob=0.8)
    scaler = GradScaler("cuda", enabled=True)

    out_dir = Path(f"checkpoints/{args.experiment_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    best_auc = 0.0
    print(f"\n{'='*60}\nDANN training: {args.experiment_name} | {args.epochs} epochs | "
          f"lambda_max={args.lambda_max}\n{'='*60}")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        spec_meter = AverageMeter()
        domain_meter = AverageMeter()
        domain_acc_meter = AverageMeter()
        for step, batch in enumerate(train_loader):
            mel, labels, domain = batch
            mel = mel.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            domain = domain.to(device, non_blocking=True)
            mel = spec_aug(mel)
            mel, labels, domain = mixup(mel, labels, domain)

            # DANN λ ramp: progress 0→1 over total_steps
            p = global_step / max(total_steps, 1)
            lambda_ = args.lambda_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

            with autocast("cuda"):
                clip_logits, _, domain_logits = model(mel, lambda_=lambda_)
                spec_loss = species_loss_fn(clip_logits, labels)
                dom_loss = domain_loss_fn(domain_logits, domain)
                loss = spec_loss + dom_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(); scheduler.step()
            global_step += 1

            spec_meter.update(spec_loss.item(), mel.size(0))
            domain_meter.update(dom_loss.item(), mel.size(0))
            with torch.no_grad():
                dom_pred = domain_logits.argmax(dim=1)
                dom_acc = (dom_pred == domain).float().mean().item()
                domain_acc_meter.update(dom_acc, mel.size(0))

            if (step + 1) % 200 == 0:
                print(f"  ep{epoch+1} step {step+1}/{len(train_loader)} "
                      f"spec={spec_meter.avg:.4f} dom_loss={domain_meter.avg:.4f} "
                      f"dom_acc={domain_acc_meter.avg:.3f} lambda={lambda_:.4f}", flush=True)

        # Eval
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for mel, lbl in val_loader:
                mel = mel.to(device)
                clip_logits, _, _ = model(mel, lambda_=0.0)
                all_p.append(torch.sigmoid(clip_logits).cpu().numpy())
                all_l.append(lbl.numpy())
        val_auc = macro_auc(np.concatenate(all_l), np.concatenate(all_p))

        sl_p, sl_l = [], []
        with torch.no_grad():
            for mel, lbl in sl_loader:
                mel = mel.to(device)
                clip_logits, _, _ = model(mel, lambda_=0.0)
                sl_p.append(torch.sigmoid(clip_logits).cpu().numpy())
                sl_l.append(lbl.numpy())
        sl_auc = macro_auc(np.concatenate(sl_l), np.concatenate(sl_p))

        is_best = sl_auc > best_auc
        if is_best:
            best_auc = sl_auc
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch + 1, "val_auc": val_auc, "sl_auc": sl_auc,
                        "label_cols": label_cols},
                       out_dir / f"best_fold{args.fold}.pt")
        torch.save({"model_state_dict": model.state_dict(),
                    "epoch": epoch + 1, "val_auc": val_auc, "sl_auc": sl_auc,
                    "label_cols": label_cols},
                   out_dir / f"epoch{epoch+1}_fold{args.fold}.pt")

        lr_now = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch+1:02d}/{args.epochs} | spec={spec_meter.avg:.4f} | "
              f"dom_loss={domain_meter.avg:.4f} | dom_acc={domain_acc_meter.avg:.3f} | "
              f"val_auc={val_auc:.4f} | sl_auc={sl_auc:.4f} | lr={lr_now:.2e}"
              f"{' *BEST*' if is_best else ''}", flush=True)

    print(f"\nBest sl_auc={best_auc:.4f}")


if __name__ == "__main__":
    main()
