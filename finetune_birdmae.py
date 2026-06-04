"""V186 Plan A: Bird-MAE-Base end-to-end finetune on BirdCLEF 2026 + V137 pseudo.

Per Rauch TMLR 08/2025: Bird-MAE backbone + classification head, layer-decay 0.75,
asymmetric loss, mel from kaldi fbank (32kHz, 128 mels, 512 frames, mean=-7.2, std=4.43).

Differences from train.py V2S finetune:
  - Backbone: Bird-MAE ViT-B (85.5M) instead of EffNetV2-S (20.8M)
  - Mel input shape: (1, 512, 128) instead of (3, 224, T)
  - Need custom mel preprocessing (kaldi fbank, not torchaudio.MelSpectrogram)
  - Single channel input (1 ch) instead of 3
  - Smaller batch (8 vs 24) due to larger model
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel

from src.config import Config
from src.dataset import build_splits, build_sampler
from src.losses import build_loss, AsymmetricLoss
from src.utils import set_seed, macro_auc, AverageMeter


SR = 32000
NUM_MELS = 128
TARGET_LEN = 512
MEAN = -7.2
STD = 4.43


def fbank_mel(waveform):
    """Bird-MAE compatible mel via kaldi fbank.

    waveform: (T,) at 32kHz
    Returns: (1, 512, 128) tensor
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    fbank = kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=SR,
        use_energy=False, window_type="hanning",
        num_mel_bins=NUM_MELS, dither=0.0, frame_shift=10,
    )
    n = fbank.shape[0]
    if n < TARGET_LEN:
        fbank = torch.cat([fbank, torch.zeros(TARGET_LEN - n, NUM_MELS)], dim=0)
    elif n > TARGET_LEN:
        fbank = fbank[:TARGET_LEN, :]
    fbank = (fbank - MEAN) / (STD * 2.0)
    return fbank.unsqueeze(0)  # (1, 512, 128)


class BirdMAEDataset(Dataset):
    def __init__(self, df, audio_dir, label_cols, target_samples=160000,
                 is_train=True, extra_audio_dirs=None):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.label_cols = label_cols
        self.label_to_idx = {l: i for i, l in enumerate(label_cols)}
        self.target_samples = target_samples
        self.is_train = is_train
        self.extra_audio_dirs = (
            {k: Path(v) for k, v in extra_audio_dirs.items()}
            if extra_audio_dirs else None
        )
        self.has_source = "source" in self.df.columns

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, row):
        if self.has_source and self.extra_audio_dirs is not None:
            src = str(row["source"])
            base = self.extra_audio_dirs.get(src, self.audio_dir)
            return Path(base) / row["filename"]
        return self.audio_dir / row["filename"]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._resolve_path(row)
        try:
            wav, sr = torchaudio.load(str(path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
            wav = wav.squeeze(0)
        except Exception:
            wav = torch.zeros(self.target_samples)

        # Random crop if training, center crop if val
        if wav.shape[0] >= self.target_samples:
            if self.is_train:
                start = torch.randint(0, wav.shape[0] - self.target_samples + 1, (1,)).item()
            else:
                start = (wav.shape[0] - self.target_samples) // 2
            wav = wav[start:start + self.target_samples]
        else:
            wav = torch.cat([wav, torch.zeros(self.target_samples - wav.shape[0])])

        mel = fbank_mel(wav)  # (1, 512, 128)

        # Multi-hot label
        label = torch.zeros(len(self.label_cols))
        primary = str(row["primary_label"])
        if primary in self.label_to_idx:
            label[self.label_to_idx[primary]] = 1.0
        return mel, label


class BirdMAEClassifier(nn.Module):
    def __init__(self, num_classes=234, dropout=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "DBD-research-group/Bird-MAE-Base", trust_remote_code=True
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        # x: (B, 1, 512, 128)
        out = self.backbone(x)
        feat = out.last_hidden_state  # (B, 768) — already mean-pooled
        feat = self.dropout(feat)
        return self.head(feat)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--train_csv", default="train.csv")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone_lr_mult", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--output_dir", default="checkpoints/birdmae_ft")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gamma_neg", type=float, default=4.0)
    ap.add_argument("--gamma_pos", type=float, default=0.0)
    ap.add_argument("--clip_margin", type=float, default=0.05)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build splits — minimal Config to satisfy build_splits
    cfg = Config()
    cfg.n_folds = args.n_folds
    cfg.fold = args.fold
    cfg.seed = args.seed
    cfg.train_csv = args.train_csv

    data_dir = Path(args.data_dir)
    train_csv_path = data_dir / args.train_csv
    df_train, df_val, label_cols = build_splits(train_csv_path, data_dir / "taxonomy.csv", cfg)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, classes: {len(label_cols)}")
    if "source" in df_train.columns:
        print(df_train["source"].value_counts())

    extra_audio_dirs = {
        "2026": data_dir / "train_audio",
        "2025": Path("data_external/2025_audio/birdclef-2025/train_audio"),
        "XC": Path("data_external/xc_pantanal_audio"),
    }

    train_ds = BirdMAEDataset(df_train, data_dir / "train_audio", label_cols,
                               target_samples=160000, is_train=True,
                               extra_audio_dirs=extra_audio_dirs)
    val_ds = BirdMAEDataset(df_val, data_dir / "train_audio", label_cols,
                             target_samples=160000, is_train=False,
                             extra_audio_dirs=extra_audio_dirs)

    print(f"Train ds: {len(train_ds)}, Val ds: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = BirdMAEClassifier(num_classes=len(label_cols)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M params")

    # Differential LR: backbone vs head
    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # Cosine schedule
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = steps_per_epoch  # 1 epoch warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = AsymmetricLoss(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg,
                              clip_margin=args.clip_margin)
    scaler = GradScaler("cuda", enabled=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    best_auc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        for step, (mel, label) in enumerate(train_loader):
            mel = mel.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.float16):
                logits = model(mel)
                loss = loss_fn(logits, label) / args.grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            meter.update(loss.item() * args.grad_accum_steps)
            if (step + 1) % 500 == 0:
                print(f"  step {step+1}/{len(train_loader)} loss={meter.avg:.4f}", flush=True)

        # Validation
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for mel, label in val_loader:
                mel = mel.to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.float16):
                    logits = model(mel)
                val_logits.append(torch.sigmoid(logits.float()).cpu().numpy())
                val_labels.append(label.numpy())
        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_auc = macro_auc(val_labels, val_logits)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[1]["lr"]
        is_best = val_auc > best_auc
        marker = " *BEST*" if is_best else ""
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss={meter.avg:.4f} | "
              f"val_auc={val_auc:.4f} | lr={lr_now:.2e} | {elapsed:.0f}s{marker}", flush=True)

        if is_best:
            best_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "epoch": epoch + 1,
                "val_auc": val_auc,
                "label_cols": label_cols,
            }, out_dir / f"best_fold{args.fold}.pt")

    print(f"\nBest: val_auc={best_auc:.4f}")


if __name__ == "__main__":
    main()
