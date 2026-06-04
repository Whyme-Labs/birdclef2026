"""
Pseudo-labeling V3 — 3-model ensemble with TTA for round 2+.

Improvements over V2:
- Supports arbitrary number of model checkpoints
- TTA with 2.5s time shift for more robust predictions
- Generates predictions on GPU for speed
"""
import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path
from src.models_v2 import SEDModelV2
import torchaudio.transforms as T


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True,
                        help="Comma-separated model checkpoint paths")
    parser.add_argument("--soundscape_dir", type=str, default="data/train_soundscapes")
    parser.add_argument("--output_dir", type=str, default="pseudo_labels_v2/round2")
    parser.add_argument("--k", type=float, default=1.0,
                        help="Threshold: mu + k*sigma")
    parser.add_argument("--theta_min", type=float, default=0.3)
    parser.add_argument("--theta_max", type=float, default=0.9)
    parser.add_argument("--tta", action="store_true",
                        help="Enable TTA with 2.5s time shift")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for GPU inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load taxonomy
    taxonomy = pd.read_csv("data/taxonomy.csv")
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())

    # Load models
    models = []
    for path in args.checkpoints.split(","):
        path = path.strip()
        model = SEDModelV2()
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device).eval()
        models.append(model)
        print(f"Loaded: {path}")
    print(f"Ensemble: {len(models)} models")

    # Mel transform
    mel_tf = T.MelSpectrogram(sample_rate=32000, n_fft=2048, hop_length=512,
                               n_mels=224, f_min=0, f_max=16000,
                               power=2.0, norm="slaney", mel_scale="htk").to(device)
    db_tf = T.AmplitudeToDB(stype="power", top_db=80.0).to(device)

    # Process soundscapes
    soundscape_dir = Path(args.soundscape_dir)
    audio_files = sorted(soundscape_dir.glob("*.ogg"))
    print(f"Processing {len(audio_files)} soundscapes (TTA={'on' if args.tta else 'off'})...")

    SR = 32000
    CHUNK = SR * 5
    HALF = CHUNK // 2
    all_records = []

    for fi, audio_path in enumerate(audio_files):
        if (fi + 1) % 200 == 0:
            print(f"  {fi+1}/{len(audio_files)}")

        audio, _ = librosa.load(audio_path, sr=SR, mono=True)
        audio = np.nan_to_num(audio, nan=0.0).astype(np.float32)
        n_chunks = max(1, len(audio) // CHUNK)
        audio = np.pad(audio, (0, max(0, n_chunks * CHUNK - len(audio))))[:n_chunks * CHUNK]

        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        with torch.no_grad():
            # Original predictions
            probs_orig = _predict_batch(
                audio.reshape(n_chunks, CHUNK), models, mel_tf, db_tf,
                device, len(label_cols), args.batch_size)

            if args.tta and n_chunks > 1:
                # Shifted predictions (+2.5s offset)
                shifted = audio[HALF:]
                n_shifted = len(shifted) // CHUNK
                if n_shifted > 0:
                    shifted = shifted[:n_shifted * CHUNK]
                    probs_shifted = _predict_batch(
                        shifted.reshape(n_shifted, CHUNK), models, mel_tf, db_tf,
                        device, len(label_cols), args.batch_size)

                    # Combine with overlap weighting
                    probs = np.zeros_like(probs_orig)
                    counts = np.zeros(n_chunks)
                    probs += 2.0 * probs_orig
                    counts += 2.0
                    for j in range(n_shifted):
                        if j < n_chunks:
                            probs[j] += probs_shifted[j]
                            counts[j] += 1.0
                        if j + 1 < n_chunks:
                            probs[j + 1] += probs_shifted[j]
                            counts[j + 1] += 1.0
                    probs /= counts[:, None]
                else:
                    probs = probs_orig
            else:
                probs = probs_orig

        for i in range(n_chunks):
            record = {"file": audio_path.name, "end_time": (i + 1) * 5.0}
            for j, col in enumerate(label_cols):
                record[col] = probs[i, j]
            all_records.append(record)

    pred_df = pd.DataFrame(all_records)
    print(f"Generated {len(pred_df)} predictions")

    # Per-class adaptive thresholding
    thresholds = {}
    for col in label_cols:
        if col in pred_df.columns:
            vals = pred_df[col].values
            mu, sigma = vals.mean(), vals.std()
            thresholds[col] = np.clip(mu + args.k * sigma, args.theta_min, args.theta_max)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "raw_predictions.csv", index=False)

    has_positive = np.zeros(len(pred_df), dtype=bool)
    for col in label_cols:
        if col in thresholds:
            has_positive |= (pred_df[col].values >= thresholds[col])
    filtered = pred_df[has_positive].copy()

    thresh_df = pd.DataFrame({"species": list(thresholds.keys()),
                                "threshold": list(thresholds.values())})
    thresh_df.to_csv(out_dir / "thresholds.csv", index=False)

    print(f"Retained: {len(filtered)}/{len(pred_df)} windows ({len(filtered)/len(pred_df)*100:.1f}%)")
    print(f"Threshold range: [{min(thresholds.values()):.3f}, {max(thresholds.values()):.3f}]")
    print(f"Saved to {out_dir}")


def _predict_batch(chunks_np, models, mel_tf, db_tf, device, num_classes, batch_size):
    """Run ensemble prediction on audio chunks with batching."""
    n = chunks_np.shape[0]
    all_probs = np.zeros((n, num_classes), dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.from_numpy(chunks_np[start:end]).float().to(device)

        # Mel spectrogram
        mel = db_tf(mel_tf(batch))
        B = mel.shape[0]
        flat = mel.reshape(B, -1)
        mel_min = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel_max = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
        mel = mel.unsqueeze(1).repeat(1, 3, 1, 1)

        # Ensemble
        for model in models:
            probs = model(mel)["clipwise_prob"].cpu().numpy()
            all_probs[start:end] += probs / len(models)

    return all_probs


if __name__ == "__main__":
    main()
