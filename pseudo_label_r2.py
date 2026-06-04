"""
Pseudo-labeling Round 2 — Hybrid PyTorch GPU + ONNX CPU ensemble.

Models:
  - EfficientNetV2-S (PyTorch on GPU): from effv2s_asl_auc_pseudo checkpoint
  - EfficientNet-B0 SED (ONNX on CPU): LB872_soup3.onnx

Mel spectrograms computed on GPU with torchaudio for speed. Both models use
the same mel: sr=32000, n_fft=2048, hop=512, n_mels=224, fmin=0, fmax=16000.

Output: pseudo_labels_r2/raw_predictions.csv
"""
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.filters
import onnxruntime as ort
import torch
import torchaudio.transforms as T
from pathlib import Path
import time
import argparse
import gc

from src.config import Config as ExperimentConfig
from src.models import SEDModel


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

# For ONNX CPU fallback mel (numpy-based, exact match with Kaggle notebook)
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
    """Numpy-based mel computation for ONNX model (CPU). Exact Kaggle notebook match."""
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
    parser = argparse.ArgumentParser(description="Round 2 pseudo-labeling with GPU + ONNX ensemble")
    parser.add_argument("--soundscape_dir", type=str, default="data/train_soundscapes")
    parser.add_argument("--output_dir", type=str, default="pseudo_labels_r2")
    parser.add_argument("--torch_ckpt", type=str,
                        default="checkpoints/effv2s_asl_auc_pseudo/best_fold0.pt",
                        help="PyTorch checkpoint for GPU model")
    parser.add_argument("--onnx_model", type=str,
                        default="kaggle_model/LB872_soup3.onnx",
                        help="ONNX model for CPU ensemble")
    parser.add_argument("--weight_torch", type=float, default=0.5)
    parser.add_argument("--weight_onnx", type=float, default=0.5)
    parser.add_argument("--k", type=float, default=1.0, help="Threshold: mu + k*sigma")
    parser.add_argument("--theta_min", type=float, default=0.3)
    parser.add_argument("--theta_max", type=float, default=0.9)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load taxonomy for label columns
    taxonomy = pd.read_csv("data/taxonomy.csv")
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())
    n_classes = len(label_cols)
    print(f"Species: {n_classes}")

    # ── Load PyTorch model (GPU) ──────────────────────────────────────────────
    print(f"\nLoading PyTorch model: {args.torch_ckpt}")
    ckpt = torch.load(args.torch_ckpt, map_location=device, weights_only=False)
    config_dict = ckpt["config"]
    config = ExperimentConfig(**{k: v for k, v in config_dict.items()
                                  if k in ExperimentConfig.__dataclass_fields__})
    print(f"  Backbone: {config.backbone}")
    torch_model = SEDModel(config).to(device)
    torch_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    torch_model.eval()
    print(f"  Val AUC: {ckpt.get('val_auc', 'N/A')}")

    # GPU mel transforms
    mel_tf = T.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=FMIN, f_max=FMAX,
        power=2.0, norm="slaney", mel_scale="htk"
    ).to(device)
    db_tf = T.AmplitudeToDB(stype="power", top_db=80.0).to(device)

    # ── Load ONNX model (CPU) ────────────────────────────────────────────────
    print(f"\nLoading ONNX model: {args.onnx_model}")
    sopts = ort.SessionOptions()
    sopts.inter_op_num_threads = 6
    sopts.intra_op_num_threads = 6
    onnx_sess = ort.InferenceSession(args.onnx_model, sopts)
    onnx_in = onnx_sess.get_inputs()[0].name
    print(f"  Input: {onnx_in}, shape: {onnx_sess.get_inputs()[0].shape}")

    # Verify shapes
    dummy_np = np.random.randn(1, 3, N_MELS, 313).astype(np.float32)
    out_onnx = onnx_sess.run(None, {onnx_in: dummy_np})[0]
    print(f"  Output shape: {out_onnx.shape}")
    assert out_onnx.shape[1] == n_classes

    # ── Process soundscapes ──────────────────────────────────────────────────
    soundscape_dir = Path(args.soundscape_dir)
    audio_files = sorted(soundscape_dir.glob("*.ogg"))
    print(f"\nProcessing {len(audio_files)} soundscapes...")

    all_records = []
    w1, w2 = args.weight_torch, args.weight_onnx
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

        # ── PyTorch model on GPU ──
        with torch.no_grad():
            batch_t = torch.from_numpy(chunks).float().to(device)
            mel = db_tf(mel_tf(batch_t))
            B = mel.shape[0]
            flat = mel.reshape(B, -1)
            mel_min = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
            mel_max = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
            mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
            mel = mel.unsqueeze(1).repeat(1, 3, 1, 1)

            clip_logits, _ = torch_model(mel)
            probs_torch = torch.sigmoid(clip_logits).cpu().numpy()

        # ── ONNX model on CPU ──
        mel_np = fast_mel_numpy(chunks)
        probs_onnx = onnx_sess.run(None, {onnx_in: mel_np})[0]

        # Weighted average
        probs = (w1 * probs_torch + w2 * probs_onnx) / w_sum

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
