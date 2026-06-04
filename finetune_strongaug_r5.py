"""
V168 R5 student — STRONG augmentation + soundscape noise injection.

The 1st place 2025 BirdCLEF recipe ("Multi-Iterative Noisy Student"):
  - Each round, train a new student with PROGRESSIVELY STRONGER augmentation
  - V137 used R2 student with mixup α=0.5, mild SpecAug
  - V168 R5 = mixup α=1.0/prob=0.8, 2x SpecAug, soundscape-noise injection (synth data)

Key change vs finetune_v2.py:
  - Mixup α=1.0 (was 0.5), prob=0.8 (was 0.5) — much stronger label mixing
  - SpecAugment: 2 freq + 2 time masks, 2x params
  - SoundscapeNoiseInjection: per-batch random mix with random noise sample at SNR ∈ [-5, 10] dB
  - Additional Gaussian white-noise injection at low probability
  - Train on focal + labeled soundscape + R4 pseudo-labeled soundscape

Usage:
  python finetune_strongaug_r5.py \\
      --pseudo_label_dir pseudo_labels_r4 \\
      --epochs 12 --lr 5e-4 \\
      --experiment_name effv2s_r5_strongaug
"""
import os, sys, time, json, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.models import SEDModel
from src.dataset import BirdCLEFDataset, SoundscapeDataset, PseudoLabelDataset
from src.losses import AsymmetricLoss
from src.utils import set_seed, macro_auc, AverageMeter
import torch.nn.functional as F


class StrongSpecAugment(nn.Module):
    """2x stronger than baseline SpecAugment."""
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


class GaussianNoiseInject(nn.Module):
    """Random additive Gaussian noise on mel spectrogram."""
    def __init__(self, std_min=0.01, std_max=0.05, prob=0.3):
        super().__init__()
        self.std_min = std_min; self.std_max = std_max; self.prob = prob

    def forward(self, mel):
        if torch.rand(1).item() > self.prob:
            return mel
        std = torch.empty(1).uniform_(self.std_min, self.std_max).item()
        noise = torch.randn_like(mel) * std
        return (mel + noise).clamp(0, 1)


class SoundscapeNoiseMixer(nn.Module):
    """
    Mix training mel with a random soundscape mel at SNR ∈ [snr_min, snr_max] dB.
    Acts as the *synthetic data* approach: focal recordings get pseudo-soundscape
    style by mixing with real soundscape ambient noise.
    """
    def __init__(self, soundscape_mel_bank, snr_min=-5.0, snr_max=10.0, prob=0.5):
        super().__init__()
        # soundscape_mel_bank: tensor (N, 1, n_mels, T) or (N, n_mels, T)
        if soundscape_mel_bank.ndim == 3:
            soundscape_mel_bank = soundscape_mel_bank.unsqueeze(1)
        self.register_buffer("bank", soundscape_mel_bank, persistent=False)
        self.snr_min = snr_min; self.snr_max = snr_max; self.prob = prob

    def forward(self, mel):
        if torch.rand(1).item() > self.prob:
            return mel
        # mel: (B, C, n_mels, T)
        B = mel.size(0)
        idx = torch.randint(0, self.bank.size(0), (B,), device=mel.device)
        noise = self.bank[idx]
        # Broadcast over channels if mel has more channels than bank
        if noise.size(1) != mel.size(1):
            noise = noise.expand(-1, mel.size(1), -1, -1)
        # Resize to match
        if noise.shape[-2:] != mel.shape[-2:]:
            noise = F.interpolate(noise, size=mel.shape[-2:], mode='bilinear', align_corners=False)
        snr_db = torch.empty(B, device=mel.device).uniform_(self.snr_min, self.snr_max)
        snr_lin = 10 ** (snr_db / 20).view(B, 1, 1, 1)
        # Mix in spectrogram domain (mel is normalized to [0, 1])
        mixed = (snr_lin * mel + noise) / (snr_lin + 1.0 + 1e-8)
        return mixed.clamp(0, 1)


class StrongMixup:
    """Mixup at higher α=1.0 with prob=0.8."""
    def __init__(self, alpha=1.0, prob=0.8):
        self.alpha = alpha; self.prob = prob

    def __call__(self, mel, labels):
        if torch.rand(1).item() > self.prob:
            return mel, labels
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(mel.size(0), device=mel.device)
        mixed_mel = lam * mel + (1 - lam) * mel[idx]
        mixed_labels = lam * labels + (1 - lam) * labels[idx]
        return mixed_mel, mixed_labels


class StrongTrainAugment(nn.Module):
    """Composition of strong augmentations."""
    def __init__(self, soundscape_mel_bank=None):
        super().__init__()
        self.spec_aug = StrongSpecAugment()
        self.gauss = GaussianNoiseInject()
        self.soundscape_mixer = (SoundscapeNoiseMixer(soundscape_mel_bank)
                                 if soundscape_mel_bank is not None else None)

    def forward(self, mel):
        # Soundscape noise mix first (in mel domain, simulates noisy field recording)
        if self.soundscape_mixer is not None:
            mel = self.soundscape_mixer(mel)
        mel = self.spec_aug(mel)
        mel = self.gauss(mel)
        return mel


def build_soundscape_mel_bank(config, n_samples=200, max_duration_s=5.0):
    """
    Pre-compute a bank of soundscape mel spectrograms. Used for synthetic noise mixing.
    Selects random 5s chunks from train_soundscapes.
    """
    print(f"Building soundscape mel bank ({n_samples} chunks)…")
    sc_dir = Path("data/train_soundscapes")
    files = sorted(sc_dir.glob("*.ogg"))
    if not files:
        print("  No soundscapes found!")
        return None
    rng = np.random.default_rng(42)
    chunks = []
    SR = 32000
    target_len = int(SR * max_duration_s)

    n_files = min(n_samples, len(files) * 12)  # 12 chunks per soundscape
    for _ in range(n_samples * 2):  # try 2x in case some fail
        if len(chunks) >= n_samples:
            break
        f = files[rng.integers(0, len(files))]
        try:
            wav, sr = sf.read(str(f))
            if sr != SR:
                continue
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if len(wav) < target_len:
                continue
            start = rng.integers(0, len(wav) - target_len)
            chunk = wav[start:start + target_len]
            chunks.append(chunk.astype(np.float32))
        except Exception:
            continue

    if not chunks:
        print("  Failed to build soundscape bank")
        return None
    print(f"  Loaded {len(chunks)} chunks")

    # Compute mel for each chunk
    mel_t = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=config.n_fft, hop_length=config.hop_length,
        n_mels=config.n_mels, f_min=config.fmin, f_max=config.fmax,
        power=2.0, norm="slaney", mel_scale="htk",
    )
    db_t = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
    bank = []
    for ch in chunks:
        wav_t = torch.from_numpy(ch).unsqueeze(0)
        mel = db_t(mel_t(wav_t))
        # min-max normalize
        mn, mx = mel.min(), mel.max()
        mel = (mel - mn) / (mx - mn + 1e-7)
        bank.append(mel)
    bank = torch.stack(bank)  # (N, 1, n_mels, T)
    print(f"  Bank shape: {bank.shape}")
    return bank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_label_dir", type=str, default="pseudo_labels_r4",
                        help="Directory with raw_predictions.csv from R4 pseudo-labeling")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="effv2s_r5_strongaug")
    parser.add_argument("--soundscape_bank_size", type=int, default=200)
    parser.add_argument("--backbone", type=str, default="tf_efficientnetv2_s.in21k")
    parser.add_argument("--warm_start_ckpt", type=str, default=None,
                        help="Optional path to warm-start checkpoint (e.g., checkpoints/effv2s_r2_fold1)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(
        backbone=args.backbone,
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
        backbone_lr_mult=0.1,
        warmup_epochs=2, loss_type="combined_asl_auc",
        n_mels=224, fmin=0.0, fmax=16000.0,
        mixup_alpha=1.0, mixup_prob=0.8,
        spec_augment=True,
        freq_mask_param=40, time_mask_param=60,
        num_freq_masks=2, num_time_masks=2,
        in_chans=3, gem_p_init=3.0, drop_rate=0.3,
        weight_decay=1e-5, max_grad_norm=1.0,
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

    # Soundscape mel bank for noise injection (synth data piece)
    sc_bank = build_soundscape_mel_bank(config, n_samples=args.soundscape_bank_size)
    if sc_bank is not None:
        sc_bank = sc_bank.to(device)
        # Augmentations applied in batch (after dataset returns mel)
        strong_aug = StrongTrainAugment(soundscape_mel_bank=sc_bank).to(device)
    else:
        strong_aug = StrongTrainAugment().to(device)
    print(f"Strong augmentation built: spec_aug + gauss_noise + "
          f"{'soundscape_mix' if sc_bank is not None else 'no_soundscape_mix'}")

    # Datasets — note: we apply strong_aug at the BATCH level (in train loop),
    # so the dataset uses minimal/no augmentation
    train_ds_focal = BirdCLEFDataset(df_train, data_dir / "train_audio", label_cols, config,
                                      is_train=True, augmentations=None)

    sl_path = data_dir / "train_soundscapes_labels.csv"
    sl_ds = SoundscapeDataset(sl_path, data_dir / "train_soundscapes",
                               label_cols, config, is_train=True, augmentations=None)
    print(f"Focal: {len(train_ds_focal)}, Soundscape: {len(sl_ds)}")

    pl_csv = Path(args.pseudo_label_dir) / "raw_predictions.csv"
    pl_ds = None
    if pl_csv.exists():
        pl_ds = PseudoLabelDataset(pl_csv, data_dir / "train_soundscapes",
                                    label_cols, config, augmentations=None,
                                    label_weight=0.5)
        print(f"Pseudo-labeled R4: {len(pl_ds)}")

    train_datasets = [train_ds_focal, sl_ds]
    if pl_ds is not None:
        train_datasets.append(pl_ds)
    train_ds = ConcatDataset(train_datasets)
    val_ds = BirdCLEFDataset(df_val, data_dir / "train_audio", label_cols, config,
                              is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    sl_eval_ds = SoundscapeDataset(sl_path, data_dir / "train_soundscapes",
                                    label_cols, config, is_train=False)
    sl_eval_loader = DataLoader(sl_eval_ds, batch_size=args.batch_size * 2,
                                 shuffle=False, num_workers=2, pin_memory=True)

    # Model — build SEDModel with chosen backbone
    config.num_classes = 234
    model = SEDModel(config).to(device)
    if args.warm_start_ckpt:
        ckpt = torch.load(args.warm_start_ckpt, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"Warm-start: missing={len(miss)} unexp={len(unexp)} from {args.warm_start_ckpt}")

    # Optimizer (SEDModel uses gem + head, not gem_pool)
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.gem.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * config.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-5)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = config.warmup_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        s / max(warmup_steps, 1) if s < warmup_steps else
        0.5 * (1 + np.cos(np.pi * (s - warmup_steps) / max(total_steps - warmup_steps, 1))))

    from src.losses import CombinedASLAUCLoss
    loss_fn = CombinedASLAUCLoss(asl_weight=0.7, auc_weight=0.3)
    mixup = StrongMixup(alpha=1.0, prob=0.8)
    scaler = GradScaler("cuda", enabled=True)

    out_dir = Path(f"checkpoints/{args.experiment_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    best_auc = 0.0
    print(f"\n{'='*60}\nTraining: {args.experiment_name} | {args.epochs} epochs | bs={args.batch_size} | "
          f"strong_aug={sc_bank is not None}\n{'='*60}")

    for epoch in range(args.epochs):
        model.train()
        meter = AverageMeter()
        for step, (mel, labels) in enumerate(train_loader):
            mel = mel.to(device); labels = labels.to(device)
            mel = strong_aug(mel)        # in-batch synthetic data + noise + spec-aug
            mel, labels = mixup(mel, labels)   # strong mixup
            with autocast("cuda"):
                clip_logits, _ = model(mel)
                loss = loss_fn(clip_logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(); scheduler.step()
            meter.update(loss.item(), mel.size(0))
            if (step + 1) % 200 == 0:
                print(f"  ep{epoch+1} step {step+1}/{len(train_loader)} loss={meter.avg:.4f}", flush=True)

        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mel, labels in val_loader:
                mel = mel.to(device)
                clip_logits, _ = model(mel)
                probs = torch.sigmoid(clip_logits).cpu().numpy()
                all_preds.append(probs); all_labels.append(labels.numpy())
        val_auc = macro_auc(np.concatenate(all_labels), np.concatenate(all_preds))

        sl_preds, sl_labels = [], []
        with torch.no_grad():
            for mel, labels in sl_eval_loader:
                mel = mel.to(device)
                clip_logits, _ = model(mel)
                probs = torch.sigmoid(clip_logits).cpu().numpy()
                sl_preds.append(probs); sl_labels.append(labels.numpy())
        sl_auc = macro_auc(np.concatenate(sl_labels), np.concatenate(sl_preds))

        select_auc = sl_auc
        is_best = select_auc > best_auc
        if is_best:
            best_auc = select_auc
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch + 1, "val_auc": val_auc, "sl_auc": sl_auc,
                        "label_cols": label_cols},
                       out_dir / f"best_fold{args.fold}.pt")
        # Always save final epoch separately
        torch.save({"model_state_dict": model.state_dict(),
                    "epoch": epoch + 1, "val_auc": val_auc, "sl_auc": sl_auc,
                    "label_cols": label_cols},
                   out_dir / f"epoch{epoch+1}_fold{args.fold}.pt")

        lr_now = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss={meter.avg:.4f} | "
              f"val_auc={val_auc:.4f} | sl_auc={sl_auc:.4f} | lr={lr_now:.2e}"
              f"{' *BEST*' if is_best else ''}", flush=True)

    print(f"\nBest sl_auc={best_auc:.4f}")


if __name__ == "__main__":
    main()
