"""
Pseudo-labeling Round 5 — Diverse-teacher triple ONNX ensemble.

Models (genuinely different architectures):
  - ConvNeXt-tiny (128-mel):          convnext_tiny_128mel.onnx
  - EfficientNetV2-S R2 (224-mel):    effv2s_r2_fold1.onnx   (val_auc=0.9454)
  - EfficientNet-B3 R2 (224-mel):     b3_softauc_r2.onnx     (val_auc=0.9410)

Per-model mel basis is selected by inspecting each ONNX session's input shape
(axis 2 = n_mels). 224-mel models: fmin=0, fmax=16000. 128-mel models: fmin=50,
fmax=14000.

Ensemble: equal weights (1/3 each).

Output: pseudo_labels_r5/raw_predictions.csv
"""
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.filters
import onnxruntime as ort
from pathlib import Path
import time
import argparse


# ── Audio / mel parameters (shared STFT settings) ─────────────────────────────
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
N_FFT = 2048
HOP_LENGTH = 512

# Mel variants (selected per model)
MEL_224 = dict(n_mels=224, fmin=0, fmax=16000)
MEL_128 = dict(n_mels=128, fmin=50, fmax=14000)

_hann_window = np.hanning(N_FFT).astype(np.float32)
_mel_basis_224 = librosa.filters.mel(
    sr=SR, n_fft=N_FFT, n_mels=MEL_224["n_mels"],
    fmin=MEL_224["fmin"], fmax=MEL_224["fmax"], htk=True, norm="slaney"
)
_mel_basis_128 = librosa.filters.mel(
    sr=SR, n_fft=N_FFT, n_mels=MEL_128["n_mels"],
    fmin=MEL_128["fmin"], fmax=MEL_128["fmax"], htk=True, norm="slaney"
)


def read_60s(path):
    """Read a 60s soundscape file, pad/trim to exactly FILE_SAMPLES."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    return y[:FILE_SAMPLES]


def _stft_power(chunks):
    """Compute |STFT|^2 once; reused by every mel variant."""
    _pad = N_FFT // 2
    specs = []
    for chunk in chunks:
        cp = np.pad(chunk, (_pad, _pad), mode="reflect")
        nf = 1 + (len(cp) - N_FFT) // HOP_LENGTH
        frames = np.lib.stride_tricks.as_strided(
            cp, (nf, N_FFT),
            (cp.strides[0] * HOP_LENGTH, cp.strides[0])
        ).copy()
        spec = np.abs(np.fft.rfft(frames * _hann_window, axis=1)) ** 2
        specs.append(spec)
    return specs  # list of (T, F) power specs


def _spec_to_input(specs, mel_basis):
    """Project power spectrogram to mel, log-compress, normalize, stack to 3ch."""
    batch = []
    for spec in specs:
        mel = mel_basis @ spec.T
        mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
        mn, mx = mel_db.min(), mel_db.max()
        batch.append((mel_db - mn) / (mx - mn + 1e-7))
    return np.repeat(np.stack(batch)[:, np.newaxis], 3, axis=1).astype(np.float32)


def fast_mel_numpy(chunks, n_mels):
    """Numpy mel spectrogram for either 224 or 128 mel bins."""
    specs = _stft_power(chunks)
    if n_mels == 224:
        return _spec_to_input(specs, _mel_basis_224)
    elif n_mels == 128:
        return _spec_to_input(specs, _mel_basis_128)
    else:
        raise ValueError(f"Unsupported n_mels: {n_mels}")


def _load_session(path, providers):
    sopts = ort.SessionOptions()
    sopts.inter_op_num_threads = 4
    sopts.intra_op_num_threads = 4
    sess = ort.InferenceSession(path, sopts, providers=providers)
    in_name = sess.get_inputs()[0].name
    in_shape = sess.get_inputs()[0].shape
    # Shape is [batch, 3, n_mels, T]; axis 2 is n_mels
    n_mels = int(in_shape[2])
    print(f"  Input: {in_name}, shape: {in_shape}, n_mels={n_mels}")
    print(f"  Providers: {sess.get_providers()}")
    return sess, in_name, n_mels


def main():
    parser = argparse.ArgumentParser(description="Round 5 pseudo-labeling: diverse-teacher triple ensemble")
    parser.add_argument("--soundscape_dir", type=str, default="data/train_soundscapes")
    parser.add_argument("--output_dir", type=str, default="pseudo_labels_r5")
    parser.add_argument("--onnx_model_1", type=str,
                        default="kaggle_model/convnext_tiny_128mel.onnx",
                        help="ONNX model 1 (ConvNeXt-tiny, 128-mel)")
    parser.add_argument("--onnx_model_2", type=str,
                        default="kaggle_model/effv2s_r2_fold1.onnx",
                        help="ONNX model 2 (EfficientNetV2-S R2, 224-mel)")
    parser.add_argument("--onnx_model_3", type=str,
                        default="kaggle_model/b3_softauc_r2.onnx",
                        help="ONNX model 3 (EfficientNet-B3 R2, 224-mel)")
    parser.add_argument("--weight_1", type=float, default=1.0 / 3)
    parser.add_argument("--weight_2", type=float, default=1.0 / 3)
    parser.add_argument("--weight_3", type=float, default=1.0 / 3)
    parser.add_argument("--k", type=float, default=1.0, help="Threshold: mu + k*sigma")
    parser.add_argument("--theta_min", type=float, default=0.3)
    parser.add_argument("--theta_max", type=float, default=0.9)
    args = parser.parse_args()

    t0 = time.time()
    print("Round 5 pseudo-labeling: diverse-teacher triple ONNX ensemble")

    # Load taxonomy
    taxonomy = pd.read_csv("data/taxonomy.csv")
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())
    n_classes = len(label_cols)
    print(f"Species: {n_classes}")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # ── Load all 3 ONNX sessions ─────────────────────────────────────────────
    print(f"\nLoading ONNX model 1: {args.onnx_model_1}")
    sess1, in1, mels1 = _load_session(args.onnx_model_1, providers)
    print(f"\nLoading ONNX model 2: {args.onnx_model_2}")
    sess2, in2, mels2 = _load_session(args.onnx_model_2, providers)
    print(f"\nLoading ONNX model 3: {args.onnx_model_3}")
    sess3, in3, mels3 = _load_session(args.onnx_model_3, providers)

    # Smoke-test each session to verify class count and mel dim
    for sess, in_name, mels, tag in [
        (sess1, in1, mels1, "model1"),
        (sess2, in2, mels2, "model2"),
        (sess3, in3, mels3, "model3"),
    ]:
        dummy = np.random.randn(1, 3, mels, 313).astype(np.float32)
        out = sess.run(None, {in_name: dummy})[0]
        assert out.shape[1] == n_classes, f"{tag} output {out.shape[1]} != {n_classes}"
        print(f"  {tag} output shape: {out.shape} OK")

    # Cache a set of which mel sizes are actually needed (dedup work)
    needed_mels = sorted({mels1, mels2, mels3})
    print(f"\nNeeded mel variants: {needed_mels}")

    # ── Process soundscapes ──────────────────────────────────────────────────
    soundscape_dir = Path(args.soundscape_dir)
    audio_files = sorted(soundscape_dir.glob("*.ogg"))
    print(f"\nProcessing {len(audio_files)} soundscapes...")

    all_records = []
    w1, w2, w3 = args.weight_1, args.weight_2, args.weight_3
    w_sum = w1 + w2 + w3

    for fi, audio_path in enumerate(audio_files):
        if (fi + 1) % 200 == 0 or fi == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed if elapsed > 0 else 0
            eta = (len(audio_files) - fi - 1) / rate if rate > 0 else 0
            print(f"  [{fi+1:5d}/{len(audio_files)}] "
                  f"{elapsed:.0f}s elapsed, {rate:.1f} files/s, ETA {eta/60:.1f}min", flush=True)

        audio = read_60s(str(audio_path))
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        chunks = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)

        # Compute each needed mel variant exactly once
        mel_inputs = {m: fast_mel_numpy(chunks, m) for m in needed_mels}

        probs_1 = sess1.run(None, {in1: mel_inputs[mels1]})[0]
        probs_2 = sess2.run(None, {in2: mel_inputs[mels2]})[0]
        probs_3 = sess3.run(None, {in3: mel_inputs[mels3]})[0]

        probs = (w1 * probs_1 + w2 * probs_2 + w3 * probs_3) / w_sum

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

    # Per-class adaptive thresholds
    thresholds = {}
    for col in label_cols:
        vals = pred_df[col].values
        mu, sigma = vals.mean(), vals.std()
        thresholds[col] = float(np.clip(mu + args.k * sigma, args.theta_min, args.theta_max))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "raw_predictions.csv", index=False)

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
