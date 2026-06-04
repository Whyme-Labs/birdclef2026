"""
Pseudo-labeling Round 3 — Dual ONNX ensemble.

Models:
  - EfficientNetV2-S R2 (ONNX): effv2s_r2_fold1.onnx (val_auc=0.9454)
  - EfficientNet-B3 R2 (ONNX): b3_softauc_r2.onnx (val_auc=0.9410)

Mel spectrograms computed with numpy (exact Kaggle notebook match).
Both models use the same mel: sr=32000, n_fft=2048, hop=512, n_mels=224, fmin=0, fmax=16000.

Output: pseudo_labels_r3/raw_predictions.csv
"""
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.filters
import onnxruntime as ort
from pathlib import Path
import time
import argparse


# ── Audio / mel parameters ────────────────────────────────────────────────────
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 224
FMIN = 0
FMAX = 16000

# Numpy-based mel (exact match with Kaggle notebook)
_hann_window = np.hanning(N_FFT).astype(np.float32)
_mel_basis = librosa.filters.mel(
    sr=SR, n_fft=N_FFT, n_mels=N_MELS,
    fmin=FMIN, fmax=FMAX, htk=True, norm="slaney"
)


def read_60s(path):
    """Read a 60s soundscape file, pad/trim to exactly FILE_SAMPLES."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    return y[:FILE_SAMPLES]


def fast_mel_numpy(chunks):
    """Numpy-based mel computation for ONNX models. Exact Kaggle notebook match."""
    _pad = N_FFT // 2
    batch = []
    for chunk in chunks:
        cp = np.pad(chunk, (_pad, _pad), mode="reflect")
        nf = 1 + (len(cp) - N_FFT) // HOP_LENGTH
        frames = np.lib.stride_tricks.as_strided(
            cp, (nf, N_FFT),
            (cp.strides[0] * HOP_LENGTH, cp.strides[0])
        ).copy()
        spec = np.abs(np.fft.rfft(frames * _hann_window, axis=1)) ** 2
        mel = _mel_basis @ spec.T
        mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
        mn, mx = mel_db.min(), mel_db.max()
        batch.append((mel_db - mn) / (mx - mn + 1e-7))
    return np.repeat(np.stack(batch)[:, np.newaxis], 3, axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Round 3 pseudo-labeling with dual ONNX ensemble")
    parser.add_argument("--soundscape_dir", type=str, default="data/train_soundscapes")
    parser.add_argument("--output_dir", type=str, default="pseudo_labels_r3")
    parser.add_argument("--onnx_model_1", type=str,
                        default="kaggle_model/effv2s_r2_fold1.onnx",
                        help="ONNX model 1 (EfficientNetV2-S R2)")
    parser.add_argument("--onnx_model_2", type=str,
                        default="kaggle_model/b3_softauc_r2.onnx",
                        help="ONNX model 2 (EfficientNet-B3 R2)")
    parser.add_argument("--weight_1", type=float, default=0.55,
                        help="Weight for model 1 (V2S, higher val_auc)")
    parser.add_argument("--weight_2", type=float, default=0.45,
                        help="Weight for model 2 (B3)")
    parser.add_argument("--k", type=float, default=1.0, help="Threshold: mu + k*sigma")
    parser.add_argument("--theta_min", type=float, default=0.3)
    parser.add_argument("--theta_max", type=float, default=0.9)
    args = parser.parse_args()

    t0 = time.time()
    print("Round 3 pseudo-labeling: dual ONNX ensemble")

    # Load taxonomy for label columns
    taxonomy = pd.read_csv("data/taxonomy.csv")
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())
    n_classes = len(label_cols)
    print(f"Species: {n_classes}")

    # ── Load ONNX model 1 ────────────────────────────────────────────────────
    print(f"\nLoading ONNX model 1: {args.onnx_model_1}")
    sopts1 = ort.SessionOptions()
    sopts1.inter_op_num_threads = 4
    sopts1.intra_op_num_threads = 4
    # Use CUDA if available for faster inference
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess1 = ort.InferenceSession(args.onnx_model_1, sopts1, providers=providers)
    in1 = sess1.get_inputs()[0].name
    print(f"  Input: {in1}, shape: {sess1.get_inputs()[0].shape}")
    print(f"  Providers: {sess1.get_providers()}")

    # Verify
    dummy = np.random.randn(1, 3, N_MELS, 313).astype(np.float32)
    out1 = sess1.run(None, {in1: dummy})[0]
    print(f"  Output shape: {out1.shape}")
    assert out1.shape[1] == n_classes, f"Model 1 output {out1.shape[1]} != {n_classes}"

    # ── Load ONNX model 2 ────────────────────────────────────────────────────
    print(f"\nLoading ONNX model 2: {args.onnx_model_2}")
    sopts2 = ort.SessionOptions()
    sopts2.inter_op_num_threads = 4
    sopts2.intra_op_num_threads = 4
    sess2 = ort.InferenceSession(args.onnx_model_2, sopts2, providers=providers)
    in2 = sess2.get_inputs()[0].name
    print(f"  Input: {in2}, shape: {sess2.get_inputs()[0].shape}")
    print(f"  Providers: {sess2.get_providers()}")

    out2 = sess2.run(None, {in2: dummy})[0]
    print(f"  Output shape: {out2.shape}")
    assert out2.shape[1] == n_classes, f"Model 2 output {out2.shape[1]} != {n_classes}"

    # ── Process soundscapes ──────────────────────────────────────────────────
    soundscape_dir = Path(args.soundscape_dir)
    audio_files = sorted(soundscape_dir.glob("*.ogg"))
    print(f"\nProcessing {len(audio_files)} soundscapes...")

    all_records = []
    w1, w2 = args.weight_1, args.weight_2
    w_sum = w1 + w2

    for fi, audio_path in enumerate(audio_files):
        if (fi + 1) % 200 == 0 or fi == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed if elapsed > 0 else 0
            eta = (len(audio_files) - fi - 1) / rate if rate > 0 else 0
            print(f"  [{fi+1:5d}/{len(audio_files)}] "
                  f"{elapsed:.0f}s elapsed, {rate:.1f} files/s, ETA {eta/60:.1f}min")

        # Read audio
        audio = read_60s(str(audio_path))

        # Peak normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        # Split into 12 windows of 5s
        chunks = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)

        # Compute mel spectrogram (shared by both models)
        mel_np = fast_mel_numpy(chunks)

        # ── ONNX model 1 ──
        probs_1 = sess1.run(None, {in1: mel_np})[0]

        # ── ONNX model 2 ──
        probs_2 = sess2.run(None, {in2: mel_np})[0]

        # Weighted average
        probs = (w1 * probs_1 + w2 * probs_2) / w_sum

        # Build records
        for i in range(N_WINDOWS):
            record = {"file": audio_path.name, "end_time": (i + 1) * 5.0}
            for j, col in enumerate(label_cols):
                record[col] = float(probs[i, j])
            all_records.append(record)

    elapsed = time.time() - t0
    print(f"\nInference done: {len(all_records)} predictions in {elapsed:.0f}s "
          f"({len(audio_files)/elapsed:.1f} files/s)")

    # Build DataFrame
    pred_df = pd.DataFrame(all_records)

    # Per-class adaptive thresholding
    thresholds = {}
    for col in label_cols:
        vals = pred_df[col].values
        mu, sigma = vals.mean(), vals.std()
        thresholds[col] = float(np.clip(mu + args.k * sigma, args.theta_min, args.theta_max))

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "raw_predictions.csv", index=False)

    # Compute stats
    has_positive = np.zeros(len(pred_df), dtype=bool)
    for col in label_cols:
        has_positive |= (pred_df[col].values >= thresholds[col])
    n_retained = has_positive.sum()

    thresh_df = pd.DataFrame({
        "species": list(thresholds.keys()),
        "threshold": list(thresholds.values())
    })
    thresh_df.to_csv(out_dir / "thresholds.csv", index=False)

    print(f"\nResults:")
    print(f"  Total windows: {len(pred_df)}")
    print(f"  Retained (has positive): {n_retained}/{len(pred_df)} ({n_retained/len(pred_df)*100:.1f}%)")
    print(f"  Threshold range: [{min(thresholds.values()):.3f}, {max(thresholds.values()):.3f}]")
    print(f"  Saved to {out_dir}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
