"""
BirdCLEF+ 2026 Inference Script.

Generates submission CSV from trained model(s).

Post-processing pipeline (from top public solutions + our additions):
────────────────────────────────────────────────────────────────────────

1. File-level max prior ("file-max leakage"):
   p̃_c(t) = p_c(t) + β · max_t' p_c(t')

   Rationale: if a species is detected with high confidence anywhere in a
   60s soundscape, there's increased prior probability it's present in
   adjacent chunks (animals don't teleport). β=0.05 is conservative.

2. Confidence-sharpened temporal smoothing:
   a) Sharpen: q_c(t) = p_c(t)^κ  (κ > 1 amplifies peaks)
   b) Smooth: q̃_c(t) = Σ_s w_s · q_c(t+s)  (Gaussian kernel)
   c) Un-sharpen: p̂_c(t) = q̃_c(t)^{1/κ}

   This preserves sharp detections while smoothing noise. Standard smoothing
   alone blurs true peaks; sharpening first protects them.

3. Multi-model blending:
   p_final = Σ_k w_k · p_k

   Blend in probability space (after sigmoid), not logit space. This is
   empirically better when models have different calibration properties.

Usage:
  python inference.py --checkpoint checkpoints/baseline/best_fold0.pt --test_dir data/test_soundscapes
  python inference.py --checkpoints ckpt1.pt,ckpt2.pt --blend_weights 0.7,0.3
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

from src.config import Config
from src.models import SEDModel


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = Config(**ckpt["config"])
    model = SEDModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    label_cols = ckpt["label_cols"]
    return model, config, label_cols


def process_soundscape(audio_path, model, config, device):
    """
    Process a single soundscape file → per-5s-window predictions.

    Returns: (n_windows, num_classes) array of probabilities
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, config.sample_rate)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels,
        f_min=config.fmin, f_max=config.fmax,
        power=2.0, norm="slaney", mel_scale="htk",
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    chunk_samples = config.target_samples
    total_samples = waveform.shape[-1]
    n_chunks = total_samples // chunk_samples

    all_probs = []
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

    if not batch_mels:
        return np.zeros((0, config.num_classes))

    # Batch inference
    batch = torch.stack(batch_mels).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        clip_logits, _ = model(batch)
        probs = torch.sigmoid(clip_logits).cpu().numpy()

    return probs


def postprocess(probs, sharpen_exp=1.5, smooth_sigma=0.7, file_max_weight=0.05):
    """
    Post-processing pipeline.

    Args:
        probs: (n_windows, num_classes) raw probabilities
        sharpen_exp: κ for confidence sharpening
        smooth_sigma: σ for Gaussian temporal smoothing
        file_max_weight: β for file-level max prior
    """
    if probs.shape[0] <= 1:
        return probs

    # 1. File-level max prior
    file_max = probs.max(axis=0, keepdims=True)
    probs = probs + file_max_weight * file_max

    # 2. Confidence-sharpened temporal smoothing
    sharpened = np.power(probs, sharpen_exp)
    smoothed = gaussian_filter1d(sharpened, sigma=smooth_sigma, axis=0)
    probs = np.power(np.maximum(smoothed, 1e-10), 1.0 / sharpen_exp)

    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True,
                        help="Comma-separated checkpoint paths")
    parser.add_argument("--blend_weights", type=str, default="",
                        help="Comma-separated blend weights (default: equal)")
    parser.add_argument("--test_dir", type=str, default="data/test_soundscapes")
    parser.add_argument("--sample_submission", type=str, default="data/sample_submission.csv")
    parser.add_argument("--output", type=str, default="submissions/submission.csv")
    parser.add_argument("--no_postprocess", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    ckpt_paths = args.checkpoints.split(",")
    models_configs = [load_model(p.strip(), device) for p in ckpt_paths]

    # Blend weights
    if args.blend_weights:
        weights = [float(w) for w in args.blend_weights.split(",")]
    else:
        weights = [1.0 / len(models_configs)] * len(models_configs)
    weights = np.array(weights) / sum(weights)

    # Get label columns from first model
    label_cols = models_configs[0][2]

    # Load sample submission for row IDs and format
    sample_sub = pd.read_csv(args.sample_submission)
    test_dir = Path(args.test_dir)
    audio_files = sorted(test_dir.glob("*.ogg"))

    if not audio_files:
        print("No test soundscapes found. Writing sample submission.")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        sample_sub.to_csv(args.output, index=False)
        return

    all_rows = []
    for audio_path in audio_files:
        soundscape_id = audio_path.stem

        # Blend predictions from all models
        blended_probs = None
        for (model, config, _), w in zip(models_configs, weights):
            probs = process_soundscape(audio_path, model, config, device)
            if blended_probs is None:
                blended_probs = w * probs
            else:
                blended_probs += w * probs

        # Post-processing
        if not args.no_postprocess:
            blended_probs = postprocess(blended_probs)

        # Build rows
        for i in range(blended_probs.shape[0]):
            end_time = (i + 1) * 5
            row_id = f"{soundscape_id}_{end_time}"
            row = {"row_id": row_id}
            for j, col in enumerate(label_cols):
                row[col] = blended_probs[i, j]
            all_rows.append(row)

    # Save submission
    sub_df = pd.DataFrame(all_rows)
    # Ensure column order matches sample submission
    sub_df = sub_df[sample_sub.columns]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(args.output, index=False)
    print(f"Submission saved: {args.output} ({len(sub_df)} rows)")


if __name__ == "__main__":
    main()
