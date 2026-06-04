"""
Pseudo-labeling with V2 models (public Perch-finetuned checkpoints).
Much better teachers than our ImageNet-trained models.
"""
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from pathlib import Path
from src.models_v2 import SEDModelV2
import torchaudio.transforms as T


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True)
    parser.add_argument("--soundscape_dir", type=str, default="data/train_soundscapes")
    parser.add_argument("--output_dir", type=str, default="pseudo_labels_v2/round1")
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--theta_min", type=float, default=0.3)
    parser.add_argument("--theta_max", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load taxonomy for label columns
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

    # Mel transform (matching public baseline exactly)
    mel_tf = T.MelSpectrogram(sample_rate=32000, n_fft=2048, hop_length=512,
                               n_mels=224, f_min=0, f_max=16000,
                               power=2.0, norm="slaney", mel_scale="htk")
    db_tf = T.AmplitudeToDB(stype="power", top_db=80.0)

    # Process soundscapes
    soundscape_dir = Path(args.soundscape_dir)
    audio_files = sorted(soundscape_dir.glob("*.ogg"))
    print(f"Processing {len(audio_files)} soundscapes...")

    chunk_samples = 32000 * 5
    all_records = []

    for fi, audio_path in enumerate(audio_files):
        if (fi + 1) % 200 == 0:
            print(f"  {fi+1}/{len(audio_files)}")

        # Load with librosa (matching public baseline)
        audio, _ = librosa.load(audio_path, sr=32000, mono=True)
        audio = np.nan_to_num(audio, nan=0.0).astype(np.float32)

        n_chunks = max(1, len(audio) // chunk_samples)
        audio = np.pad(audio, (0, max(0, n_chunks * chunk_samples - len(audio))))[:n_chunks * chunk_samples]

        # Peak normalization (matching public baseline)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        chunks = torch.from_numpy(audio.reshape(n_chunks, chunk_samples)).float()

        # Mel spectrogram
        with torch.no_grad():
            mel = db_tf(mel_tf(chunks))
            B = mel.shape[0]
            flat = mel.reshape(B, -1)
            mel_min = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
            mel_max = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
            mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
            mel = mel.unsqueeze(1).repeat(1, 3, 1, 1).to(device)

        # Ensemble prediction
        ensemble_probs = np.zeros((n_chunks, len(label_cols)), dtype=np.float64)
        with torch.no_grad():
            for model in models:
                probs = model(mel)["clipwise_prob"].cpu().numpy()
                ensemble_probs += probs / len(models)

        for i in range(n_chunks):
            record = {"file": audio_path.name, "end_time": (i + 1) * 5.0}
            for j, col in enumerate(label_cols):
                record[col] = ensemble_probs[i, j]
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

    # Save raw predictions
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "raw_predictions.csv", index=False)

    # Filter and save
    has_positive = np.zeros(len(pred_df), dtype=bool)
    for col in label_cols:
        if col in thresholds:
            has_positive |= (pred_df[col].values >= thresholds[col])
    filtered = pred_df[has_positive].copy()

    # Save thresholds
    thresh_df = pd.DataFrame({"species": list(thresholds.keys()), "threshold": list(thresholds.values())})
    thresh_df.to_csv(out_dir / "thresholds.csv", index=False)

    print(f"Retained: {len(filtered)}/{len(pred_df)} windows ({len(filtered)/len(pred_df)*100:.1f}%)")
    print(f"Threshold range: [{min(thresholds.values()):.3f}, {max(thresholds.values()):.3f}]")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
