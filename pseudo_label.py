"""
Pseudo-Labeling Pipeline for BirdCLEF+ 2026.

This is the MOST IMPORTANT differentiator.

Uses the B0+B3 teacher ensemble to predict on all 10,658 train soundscapes,
then filters with per-class adaptive thresholds to generate soft pseudo-labels.

Scientific basis (Lasseck, BirdCLEF 2024): iterative pseudo-labeling on
target-location recordings "significantly improved performance."
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from collections import Counter

from src.config import Config
from src.models import SEDModel


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = SEDModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt["label_cols"]


def predict_soundscapes(models_cfgs, soundscape_dir, device):
    """Run ensemble on all soundscape files."""
    soundscape_dir = Path(soundscape_dir)
    audio_files = sorted(soundscape_dir.glob("*.ogg"))
    print(f"Found {len(audio_files)} soundscape files")

    # Use config from first model
    config = models_cfgs[0][1]
    label_cols = models_cfgs[0][2]
    chunk_samples = config.target_samples

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels,
        f_min=config.fmin, f_max=config.fmax,
        power=2.0, norm="slaney", mel_scale="htk",
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    all_records = []
    for fi, audio_path in enumerate(audio_files):
        if (fi + 1) % 100 == 0:
            print(f"  Processing {fi+1}/{len(audio_files)}: {audio_path.name}")

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != config.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, config.sample_rate)

        n_chunks = waveform.shape[-1] // chunk_samples
        if n_chunks == 0:
            continue

        # Build mel batch
        batch_mels = []
        for i in range(n_chunks):
            start = i * chunk_samples
            chunk = waveform[..., start:start + chunk_samples]
            mel = mel_transform(chunk)
            mel = db_transform(mel)
            mel_min, mel_max = mel.min(), mel.max()
            mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)
            mel = mel.expand(3, -1, -1)
            batch_mels.append(mel)

        batch = torch.stack(batch_mels).to(device)

        # Ensemble: average probabilities
        ensemble_probs = None
        for model, cfg, _ in models_cfgs:
            with torch.no_grad(), torch.amp.autocast("cuda"):
                clip_logits, _ = model(batch)
                probs = torch.sigmoid(clip_logits).cpu().numpy()
            if ensemble_probs is None:
                ensemble_probs = probs / len(models_cfgs)
            else:
                ensemble_probs += probs / len(models_cfgs)

        # Store
        for i in range(n_chunks):
            record = {
                "file": audio_path.name,
                "end_time": (i + 1) * 5.0,
            }
            for j, col in enumerate(label_cols):
                record[col] = ensemble_probs[i, j]
            all_records.append(record)

    return pd.DataFrame(all_records), label_cols


def filter_pseudo_labels(pred_df, label_cols, k=1.0, theta_min=0.2,
                         theta_max=0.9, min_max_prob=0.2):
    """
    Per-class adaptive thresholding.

    θ_c = clip(μ_c + k·σ_c, θ_min, θ_max)
    """
    # Compute per-class thresholds
    thresholds = {}
    for col in label_cols:
        vals = pred_df[col].values
        mu, sigma = vals.mean(), vals.std()
        theta = np.clip(mu + k * sigma, theta_min, theta_max)
        thresholds[col] = theta

    # Sample-level filter
    max_probs = pred_df[label_cols].max(axis=1)
    valid = max_probs >= min_max_prob
    filtered = pred_df[valid].copy()

    # Zero out below-threshold (soft labels: keep probability, zero out weak ones)
    for col in label_cols:
        below = filtered[col] < thresholds[col]
        filtered.loc[below, col] = 0.0

    # Remove rows with no positive
    has_pos = filtered[label_cols].max(axis=1) > 0
    filtered = filtered[has_pos]

    return filtered, thresholds


def save_pseudo_labels(pseudo_df, label_cols, output_dir):
    """Save pseudo-labels as CSV for training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build training-ready CSV
    records = []
    for _, row in pseudo_df.iterrows():
        # Find primary label (highest prob species)
        probs = {col: row[col] for col in label_cols if row[col] > 0}
        if not probs:
            continue
        primary = max(probs, key=probs.get)
        records.append({
            "filename": f"train_soundscapes_chunks/{row['file'].replace('.ogg', '')}_{int(row['end_time'])}.ogg",
            "primary_label": primary,
            "secondary_labels": "[]",
            "source_file": row["file"],
            "end_time": row["end_time"],
            "pseudo_confidence": max(probs.values()),
            **{f"soft_{col}": row[col] for col in label_cols},
        })

    result_df = pd.DataFrame(records)
    result_df.to_csv(output_dir / "pseudo_labels.csv", index=False)

    # Also save the raw predictions for analysis
    pseudo_df.to_csv(output_dir / "raw_predictions.csv", index=False)

    # Stats
    species_counts = result_df["primary_label"].value_counts()
    print(f"\nPseudo-label stats:")
    print(f"  Total windows: {len(result_df)}")
    print(f"  Species covered: {len(species_counts)}")
    print(f"  Top 10 species:")
    for sp, cnt in species_counts.head(10).items():
        print(f"    {sp}: {cnt}")
    print(f"  Bottom 10 species:")
    for sp, cnt in species_counts.tail(10).items():
        print(f"    {sp}: {cnt}")

    return result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "stats"])
    parser.add_argument("--checkpoints", type=str, default="")
    parser.add_argument("--soundscape_dir", type=str, default="data/train_soundscapes")
    parser.add_argument("--output_dir", type=str, default="pseudo_labels/round1")
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--theta_min", type=float, default=0.2)
    parser.add_argument("--theta_max", type=float, default=0.9)
    parser.add_argument("--min_max_prob", type=float, default=0.2)
    parser.add_argument("--soft", action="store_true", default=True)
    args = parser.parse_args()

    if args.command == "generate":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        ckpt_paths = [p.strip() for p in args.checkpoints.split(",")]
        models_cfgs = []
        for p in ckpt_paths:
            m, c, lc = load_model(p, device)
            models_cfgs.append((m, c, lc))
            print(f"Loaded: {p}")

        # Predict
        print(f"\nPredicting on {args.soundscape_dir}...")
        pred_df, label_cols = predict_soundscapes(models_cfgs, args.soundscape_dir, device)
        print(f"Generated {len(pred_df)} predictions")

        # Filter
        pseudo_df, thresholds = filter_pseudo_labels(
            pred_df, label_cols, k=args.k,
            theta_min=args.theta_min, theta_max=args.theta_max,
            min_max_prob=args.min_max_prob
        )
        print(f"After filtering: {len(pseudo_df)} windows "
              f"({len(pseudo_df)/len(pred_df)*100:.1f}% retained)")
        print(f"Threshold range: [{min(thresholds.values()):.3f}, {max(thresholds.values()):.3f}]")

        # Save
        save_pseudo_labels(pseudo_df, label_cols, args.output_dir)
        print(f"\nSaved to {args.output_dir}")

    elif args.command == "stats":
        df = pd.read_csv(Path(args.output_dir) / "pseudo_labels.csv")
        print(f"Pseudo-labels: {len(df)} samples")
        print(f"Species: {df['primary_label'].nunique()}")
        print(df["primary_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
