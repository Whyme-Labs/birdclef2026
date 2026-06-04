"""
Audio MAE Pretraining on Pantanal Soundscapes.

Self-supervised pretraining using masked spectrogram reconstruction.
Uses ALL audio: 10,658 unlabeled soundscapes + 35,549 training recordings.

This learns Pantanal-specific acoustic representations without labels,
bridging the domain gap between focal recordings and soundscapes.

Usage:
    python pretrain_mae.py --epochs 100 --batch_size 64 --lr 1.5e-4
"""
import os, sys, time, glob, random
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path

from src.audio_mae import AudioMAE


# ── Dataset: loads random 5s chunks from any audio file ─────────────────
class AudioChunkDataset(Dataset):
    """
    Loads random 5-second chunks from audio files.
    No labels needed — pure self-supervised.
    """
    def __init__(self, audio_files, sr=32000, chunk_duration=5.0,
                 chunks_per_file=1, n_mels=224, n_fft=2048, hop_length=512,
                 fmin=0, fmax=16000, target_width=320):
        self.files = audio_files
        self.sr = sr
        self.chunk_samples = int(sr * chunk_duration)
        self.chunks_per_file = chunks_per_file
        self.target_width = target_width

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,
            norm="slaney", mel_scale="htk",
        )
        self.db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)

    def __len__(self):
        return len(self.files) * self.chunks_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.chunks_per_file
        path = self.files[file_idx]

        try:
            info = torchaudio.info(path)
            num_frames = info.num_frames
            sr_file = info.sample_rate

            # Random start position
            if num_frames > int(self.chunk_samples * sr_file / self.sr):
                max_start = num_frames - int(self.chunk_samples * sr_file / self.sr)
                start = random.randint(0, max_start)
            else:
                start = 0

            waveform, sr_loaded = torchaudio.load(path, frame_offset=start,
                                                    num_frames=int(self.chunk_samples * sr_file / self.sr))

            # Resample if needed
            if sr_loaded != self.sr:
                resampler = T.Resample(sr_loaded, self.sr)
                waveform = resampler(waveform)

            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Pad/trim to exact chunk length
            if waveform.shape[1] < self.chunk_samples:
                waveform = torch.nn.functional.pad(waveform, (0, self.chunk_samples - waveform.shape[1]))
            waveform = waveform[:, :self.chunk_samples]

            # Peak normalize
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        except Exception as e:
            # Return silence on error
            waveform = torch.zeros(1, self.chunk_samples)

        # Mel spectrogram
        mel = self.mel_transform(waveform)  # (1, n_mels, time)
        mel = self.db_transform(mel)

        # Min-max normalize to [0, 1]
        mel_min = mel.min()
        mel_max = mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)

        # Pad/trim width to target_width for consistent patch count
        if mel.shape[-1] < self.target_width:
            mel = torch.nn.functional.pad(mel, (0, self.target_width - mel.shape[-1]))
        mel = mel[:, :, :self.target_width]

        return mel  # (1, 224, 320)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--encoder_dim", type=int, default=384)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--decoder_dim", type=int, default=192)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--chunks_per_file", type=int, default=4,
                        help="Random chunks to sample per file per epoch")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Gather all audio files ──────────────────────────────────────────
    data_dir = Path("data")
    audio_files = []

    # Unlabeled soundscapes (target domain — most important)
    sl_files = sorted(glob.glob(str(data_dir / "train_soundscapes" / "*.ogg")))
    print(f"Unlabeled soundscapes: {len(sl_files)}")
    audio_files.extend(sl_files)

    # Training recordings (source domain)
    train_files = sorted(glob.glob(str(data_dir / "train_audio" / "*" / "*.ogg")))
    print(f"Training recordings: {len(train_files)}")
    audio_files.extend(train_files)

    print(f"Total audio files: {len(audio_files)}")

    # ── Dataset and loader ──────────────────────────────────────────────
    target_width = (args.patch_size * 20)  # 320 for patch_size=16
    dataset = AudioChunkDataset(
        audio_files,
        chunks_per_file=args.chunks_per_file,
        target_width=target_width,
    )
    print(f"Dataset size: {len(dataset)} chunks/epoch")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
        drop_last=True,
    )

    # ── Model ───────────────────────────────────────────────────────────
    model = AudioMAE(
        img_size=(224, target_width),
        patch_size=args.patch_size,
        in_chans=1,
        encoder_embed_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_num_heads=args.encoder_dim // 64,
        decoder_embed_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_dim // 64,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Audio MAE: {n_params:.1f}M params")
    print(f"  Encoder: {args.encoder_depth} layers, {args.encoder_dim} dim")
    print(f"  Decoder: {args.decoder_depth} layers, {args.decoder_dim} dim")
    print(f"  Patches: {model.patch_embed.grid_size} = {model.patch_embed.num_patches}")
    print(f"  Mask ratio: {args.mask_ratio}")

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(loader)
    warmup_steps = args.warmup_epochs * len(loader)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    # ── Resume ──────────────────────────────────────────────────────────
    start_epoch = 0
    out_dir = Path("checkpoints/audio_mae")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Audio MAE Pretraining | {args.epochs} epochs | {len(loader)} steps/epoch")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        t0 = time.time()

        for batch_idx, mel in enumerate(loader):
            mel = mel.to(device)  # (B, 1, 224, 320)

            with autocast("cuda", enabled=device.type == "cuda"):
                loss, _, _ = model(mel)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            losses.append(loss.item())

            if (batch_idx + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  [{batch_idx+1}/{len(loader)}] loss={avg_loss:.4f} lr={lr_now:.2e}")

        epoch_loss = np.mean(losses)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:03d}/{args.epochs} | loss={epoch_loss:.4f} | "
              f"lr={lr_now:.2e} | {elapsed:.0f}s")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "args": vars(args),
            }, out_dir / f"mae_epoch{epoch+1:03d}.pt")
            print(f"  Saved checkpoint")

        # Always save latest
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "args": vars(args),
        }, out_dir / "mae_latest.pt")

    print(f"\nPretraining complete. Best loss: {min(losses):.4f}")


if __name__ == "__main__":
    main()
