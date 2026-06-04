"""
Finetune AudioMAE (pretrained on AudioSet-2M) for BirdCLEF 2026.

This uses the AudioMAE ViT-Base backbone from HuggingFace, pretrained on
2M audio clips. We finetune it on BirdCLEF labeled data with:
- Mel config matching AudioMAE expectations (128 mels, different hop)
- Differential LR (backbone 0.01× head)
- ASL loss for multi-label
- Optional: continue self-supervised pretraining on Pantanal soundscapes first

Usage:
    python finetune_audiomae.py --epochs 20 --lr 5e-4 --batch_size 32
"""
import os, sys, time, math, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import timm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.losses import AsymmetricLoss
from src.augmentations import Mixup
from src.utils import set_seed, macro_auc, AverageMeter


# ── AudioMAE Classifier ────────────────────────────────────────────────
class AudioMAEClassifier(nn.Module):
    """
    AudioMAE backbone + classification head.
    Uses timm's ViT with AudioMAE pretrained weights.
    """
    def __init__(self, num_classes=234, dropout=0.2, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model(
            'hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m',
            pretrained=True, num_classes=0, dynamic_img_size=True,
        )
        self.embed_dim = self.backbone.embed_dim  # 768

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)  # (B, 768) global avg pool
        logits = self.head(features)
        return {
            "clipwise_logit": logits,
            "clipwise_prob": torch.sigmoid(logits),
        }


# ── Dataset with AudioMAE-compatible mel ────────────────────────────────
class AudioMAEDataset(Dataset):
    """
    Dataset producing mel spectrograms compatible with AudioMAE.
    AudioMAE expects: (B, 1, time_frames, 128) where time_frames varies.
    We use 128 mel bins, 32kHz, hop_length=500 → 320 frames for 5s.
    """
    def __init__(self, df, audio_dir, label_cols,
                 sr=32000, chunk_duration=5.0, n_mels=128,
                 n_fft=2048, hop_length=500, fmin=20, fmax=16000,
                 is_train=True):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.label_cols = label_cols
        self.label_to_idx = {l: i for i, l in enumerate(label_cols)}
        self.sr = sr
        self.chunk_samples = int(sr * chunk_duration)
        self.is_train = is_train
        self.n_classes = len(label_cols)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,
        )
        self.db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.audio_dir / str(row["primary_label"]) / row["filename"]
        if not path.exists():
            path = self.audio_dir / row["filename"]

        try:
            waveform, sr = torchaudio.load(path)
            if sr != self.sr:
                waveform = T.Resample(sr, self.sr)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if waveform.shape[1] > self.chunk_samples:
                if self.is_train:
                    start = torch.randint(0, waveform.shape[1] - self.chunk_samples, (1,)).item()
                else:
                    start = (waveform.shape[1] - self.chunk_samples) // 2
                waveform = waveform[:, start:start + self.chunk_samples]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, self.chunk_samples - waveform.shape[1]))

            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        except Exception:
            waveform = torch.zeros(1, self.chunk_samples)

        # Mel spectrogram: (1, 128, time)
        mel = self.db_transform(self.mel_transform(waveform))
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)

        # AudioMAE expects (1, time, freq) — transpose!
        mel = mel.permute(0, 2, 1)  # (1, time, 128)

        # Ensure time dim is divisible by patch_size=16
        T_dim = mel.shape[1]
        target_T = (T_dim // 16) * 16  # Round down to nearest multiple of 16
        mel = mel[:, :target_T, :]

        # Labels
        label = torch.zeros(self.n_classes)
        primary = str(row["primary_label"])
        if primary in self.label_to_idx:
            label[self.label_to_idx[primary]] = 1.0
        if "secondary_labels" in row and pd.notna(row["secondary_labels"]):
            try:
                for sl in ast.literal_eval(str(row["secondary_labels"])):
                    sl = str(sl)
                    if sl in self.label_to_idx:
                        label[self.label_to_idx[sl]] = 1.0
            except:
                pass

        return mel, label


class AudioMAESoundscapeDataset(Dataset):
    """Soundscape dataset for AudioMAE."""
    def __init__(self, csv_path, audio_dir, label_cols,
                 sr=32000, chunk_duration=5.0, n_mels=128,
                 n_fft=2048, hop_length=500, fmin=20, fmax=16000,
                 is_train=True):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = Path(audio_dir)
        self.label_cols = label_cols
        self.label_to_idx = {l: i for i, l in enumerate(label_cols)}
        self.sr = sr
        self.chunk_samples = int(sr * chunk_duration)
        self.is_train = is_train
        self.n_classes = len(label_cols)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,
        )
        self.db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.audio_dir / row["filename"]

        start_parts = str(row["start"]).split(":")
        start_sec = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])
        start_sample = start_sec * self.sr

        try:
            waveform, sr = torchaudio.load(path, frame_offset=start_sample,
                                            num_frames=self.chunk_samples)
            if sr != self.sr:
                waveform = T.Resample(sr, self.sr)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if waveform.shape[1] < self.chunk_samples:
                waveform = torch.nn.functional.pad(waveform, (0, self.chunk_samples - waveform.shape[1]))
            waveform = waveform[:, :self.chunk_samples]
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak
        except Exception:
            waveform = torch.zeros(1, self.chunk_samples)

        mel = self.db_transform(self.mel_transform(waveform))
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
        mel = mel.permute(0, 2, 1)  # (1, time, 128)

        # Ensure time dim is divisible by patch_size=16
        T_dim = mel.shape[1]
        target_T = (T_dim // 16) * 16
        mel = mel[:, :target_T, :]

        label = torch.zeros(self.n_classes)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in self.label_to_idx:
                label[self.label_to_idx[sp]] = 1.0

        return mel, label


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--freeze_epochs", type=int, default=2,
                        help="Freeze backbone for first N epochs (linear probe)")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--experiment_name", type=str, default="audiomae_ft")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    train_ds = AudioMAEDataset(df_train, data_dir / "train_audio", label_cols, is_train=True)

    sl_path = data_dir / "train_soundscapes_labels.csv"
    if sl_path.exists():
        sl_ds = AudioMAESoundscapeDataset(sl_path, data_dir / "train_soundscapes",
                                           label_cols, is_train=True)
        train_ds = ConcatDataset([train_ds, sl_ds])
        print(f"Added {len(sl_ds)} soundscape windows")

    val_ds = AudioMAEDataset(df_val, data_dir / "train_audio", label_cols, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    # Model
    model = AudioMAEClassifier(num_classes=num_classes, dropout=args.dropout)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"AudioMAE Classifier: {n_params:.1f}M params")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Optimizer
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=0.05)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = 2 * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        s / max(warmup_steps, 1) if s < warmup_steps else
        0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(total_steps - warmup_steps, 1))))

    loss_fn = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip_margin=0.05)
    mixup = Mixup(alpha=0.3, prob=0.5)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path(f"checkpoints/{args.experiment_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    print(f"\n{'='*60}")
    print(f"AudioMAE Finetuning | {args.epochs} epochs | lr={args.lr}")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        # Freeze backbone for first N epochs
        if epoch < args.freeze_epochs:
            for p in backbone_params:
                p.requires_grad = False
            frozen_str = " (backbone frozen)"
        elif epoch == args.freeze_epochs:
            for p in backbone_params:
                p.requires_grad = True
            frozen_str = " (unfreezing backbone)"
        else:
            frozen_str = ""

        model.train()
        meter = AverageMeter()
        n_steps = len(train_loader)
        for step, (mel, labels) in enumerate(train_loader):
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
            if (step + 1) % 200 == 0:
                print(f"  ep{epoch+1} step {step+1}/{n_steps} loss={meter.avg:.4f}", flush=True)

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
              f"{' *BEST*' if is_best else ''}{frozen_str}")

    print(f"\nBest: val_auc={best_auc:.4f}")


if __name__ == "__main__":
    main()
