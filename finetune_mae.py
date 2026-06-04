"""
Finetune pretrained Audio MAE for BirdCLEF classification.

Takes the encoder from pretrained Audio MAE, attaches classification head,
and finetuned on labeled data. The encoder representations already understand
Pantanal acoustic structure from self-supervised pretraining.

Usage:
    python finetune_mae.py --mae_checkpoint checkpoints/audio_mae/mae_latest.pt --epochs 30
"""
import os, sys, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.audio_mae import AudioMAE, AudioMAEClassifier
from src.losses import AsymmetricLoss
from src.utils import set_seed, macro_auc, AverageMeter


class MAEDataset(Dataset):
    """Dataset that produces single-channel mel spectrograms for MAE classifier."""
    def __init__(self, df, audio_dir, label_cols, sr=32000, chunk_duration=5.0,
                 n_mels=224, n_fft=2048, hop_length=512, fmin=0, fmax=16000,
                 target_width=320, is_train=True):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.label_cols = label_cols
        self.label_to_idx = {l: i for i, l in enumerate(label_cols)}
        self.sr = sr
        self.chunk_samples = int(sr * chunk_duration)
        self.target_width = target_width
        self.is_train = is_train
        self.n_classes = len(label_cols)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,
            norm="slaney", mel_scale="htk",
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

            # Random crop (train) or center crop (val)
            if waveform.shape[1] > self.chunk_samples:
                if self.is_train:
                    start = torch.randint(0, waveform.shape[1] - self.chunk_samples, (1,)).item()
                else:
                    start = (waveform.shape[1] - self.chunk_samples) // 2
                waveform = waveform[:, start:start + self.chunk_samples]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, self.chunk_samples - waveform.shape[1]))

            # Peak normalize
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        except Exception:
            waveform = torch.zeros(1, self.chunk_samples)

        # Mel spectrogram
        mel = self.db_transform(self.mel_transform(waveform))
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
        if mel.shape[-1] < self.target_width:
            mel = torch.nn.functional.pad(mel, (0, self.target_width - mel.shape[-1]))
        mel = mel[:, :, :self.target_width]

        # Labels
        label = torch.zeros(self.n_classes)
        primary = str(row["primary_label"])
        if primary in self.label_to_idx:
            label[self.label_to_idx[primary]] = 1.0
        if "secondary_labels" in row and pd.notna(row["secondary_labels"]):
            try:
                import ast
                for sl in ast.literal_eval(str(row["secondary_labels"])):
                    sl = str(sl)
                    if sl in self.label_to_idx:
                        label[self.label_to_idx[sl]] = 1.0
            except:
                pass

        return mel, label


class MAESoundscapeDataset(Dataset):
    """Soundscape dataset for MAE classifier."""
    def __init__(self, csv_path, audio_dir, label_cols, sr=32000, chunk_duration=5.0,
                 n_mels=224, n_fft=2048, hop_length=512, fmin=0, fmax=16000,
                 target_width=320, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = Path(audio_dir)
        self.label_cols = label_cols
        self.label_to_idx = {l: i for i, l in enumerate(label_cols)}
        self.sr = sr
        self.chunk_samples = int(sr * chunk_duration)
        self.target_width = target_width
        self.is_train = is_train
        self.n_classes = len(label_cols)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,
            norm="slaney", mel_scale="htk",
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
        if mel.shape[-1] < self.target_width:
            mel = torch.nn.functional.pad(mel, (0, self.target_width - mel.shape[-1]))
        mel = mel[:, :, :self.target_width]

        label = torch.zeros(self.n_classes)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in self.label_to_idx:
                label[self.label_to_idx[sp]] = 1.0

        return mel, label


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mae_checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--pool", type=str, default="avg", choices=["avg", "cls"])
    parser.add_argument("--freeze_epochs", type=int, default=3,
                        help="Freeze encoder for first N epochs (linear probe)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load MAE checkpoint ─────────────────────────────────────────────
    ckpt = torch.load(args.mae_checkpoint, map_location="cpu", weights_only=False)
    mae_args = ckpt.get("args", {})
    target_width = mae_args.get("patch_size", 16) * 20  # 320

    mae = AudioMAE(
        img_size=(224, target_width),
        patch_size=mae_args.get("patch_size", 16),
        in_chans=1,
        encoder_embed_dim=mae_args.get("encoder_dim", 384),
        encoder_depth=mae_args.get("encoder_depth", 12),
        encoder_num_heads=mae_args.get("encoder_dim", 384) // 64,
        decoder_embed_dim=mae_args.get("decoder_dim", 192),
        decoder_depth=mae_args.get("decoder_depth", 4),
        decoder_num_heads=mae_args.get("decoder_dim", 192) // 64,
    )
    mae.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded MAE from epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f}")

    # ── Data ────────────────────────────────────────────────────────────
    data_dir = Path("data")
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())
    num_classes = len(label_cols)

    df = pd.read_csv(data_dir / "train.csv")
    df["primary_label"] = df["primary_label"].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(df, df["primary_label"]))[args.fold]
    df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]

    train_ds = MAEDataset(df_train, data_dir / "train_audio", label_cols,
                           target_width=target_width, is_train=True)

    # Add soundscapes
    sl_path = data_dir / "train_soundscapes_labels.csv"
    if sl_path.exists():
        sl_ds = MAESoundscapeDataset(sl_path, data_dir / "train_soundscapes", label_cols,
                                      target_width=target_width, is_train=True)
        train_ds = ConcatDataset([train_ds, sl_ds])
        print(f"Added {len(sl_ds)} soundscape windows")

    val_ds = MAEDataset(df_val, data_dir / "train_audio", label_cols,
                         target_width=target_width, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    # ── Build classifier ────────────────────────────────────────────────
    model = AudioMAEClassifier(mae, num_classes=num_classes, pool=args.pool, dropout=0.1)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MAE Classifier: {n_params:.1f}M params")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Optimizer: differential LR ──────────────────────────────────────
    encoder_params = list(model.patch_embed.parameters()) + \
                     list(model.encoder_blocks.parameters()) + \
                     list(model.encoder_norm.parameters()) + \
                     [model.cls_token, model.pos_embed]
    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=0.05)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = 3 * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        s / max(warmup_steps, 1) if s < warmup_steps else
        0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(total_steps - warmup_steps, 1))))

    loss_fn = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip_margin=0.05)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path("checkpoints/mae_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ────────────────────────────────────────────────────────
    best_auc = 0.0
    print(f"\n{'='*60}")
    print(f"MAE Classifier Finetuning | {args.epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        # Optionally freeze encoder for initial epochs
        if epoch < args.freeze_epochs:
            for p in encoder_params:
                p.requires_grad = False
        elif epoch == args.freeze_epochs:
            for p in encoder_params:
                p.requires_grad = True
            print(f"  Unfreezing encoder at epoch {epoch+1}")

        model.train()
        meter = AverageMeter()
        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)
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
                "mae_args": mae_args,
                "epoch": epoch + 1,
                "val_auc": val_auc,
                "label_cols": label_cols,
                "pool": args.pool,
            }, out_dir / f"best_fold{args.fold}.pt")

        lr_now = optimizer.param_groups[1]["lr"]
        frozen = " (encoder frozen)" if epoch < args.freeze_epochs else ""
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss={meter.avg:.4f} | "
              f"val_auc={val_auc:.4f} | lr={lr_now:.2e}"
              f"{' *BEST*' if is_best else ''}{frozen}")

    print(f"\nBest: val_auc={best_auc:.4f}")


if __name__ == "__main__":
    main()
