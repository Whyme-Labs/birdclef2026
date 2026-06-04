"""V180 Phase 1: V137 ensemble pseudo-label generation on train_soundscapes.

Following BirdCLEF 2025 1st-place "Multi-Iterative Noisy Student" recipe:
  1. Run V137 4-model ensemble on train_soundscapes (10K files)
  2. Apply PowerTransform sharpening (probs → probs^k, normalized)
  3. Save as new pseudo_labels_v137/raw_predictions.csv (compatible with train.py format)

The 4 models all use ONNX (CPU) for portability. Local GPU not used since ONNX
runtime CPU is plenty fast for this batch job (~3-5 hours total).

Output format matches pseudo_labels_r2/raw_predictions.csv:
  file, end_time, <234 species probs>
"""
import argparse, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
import soundfile as sf
import librosa
from scipy.ndimage import gaussian_filter1d


def setup_perch(model_dir):
    p = Path(model_dir) / "perch_v2.onnx"
    if not p.exists():
        # Fallback search
        for root in [Path(model_dir), Path("kaggle_model"), Path("/kaggle/input")]:
            for f in root.rglob("perch_v2.onnx"):
                p = f; break
            if p.exists(): break
    sopts = ort.SessionOptions()
    sopts.inter_op_num_threads = 4; sopts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(p), sopts, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def setup_sed(model_dir):
    p = Path(model_dir) / "LB872_soup3.onnx"
    sopts = ort.SessionOptions()
    sopts.inter_op_num_threads = 4; sopts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(p), sopts, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def setup_nfnet(model_dir):
    p = Path(model_dir) / "eca_nfnet_l0.onnx"
    sopts = ort.SessionOptions()
    sopts.inter_op_num_threads = 4; sopts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(p), sopts, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def setup_v2s(model_dir, fold=1):
    p = Path(model_dir) / f"effv2s_r2_fold{fold}.onnx"
    sopts = ort.SessionOptions()
    sopts.inter_op_num_threads = 4; sopts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(p), sopts, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


SR = 32000
WINDOW_SAMPLES = 5 * SR  # 5s at 32kHz
N_WINDOWS = 12  # 60s / 5s


def read_60s(path):
    """Read first 60s @ 32kHz mono."""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    if audio.shape[0] < N_WINDOWS * WINDOW_SAMPLES:
        audio = np.pad(audio, (0, N_WINDOWS * WINDOW_SAMPLES - audio.shape[0]))
    return audio[:N_WINDOWS * WINDOW_SAMPLES].astype(np.float32)


def fast_mel(chunks, n_mels=224, n_fft=2048, hop=512, fmin=0, fmax=16000):
    """V2S/SED mel: 224 mels @ 32kHz, output (B, 3, 224, 313)."""
    mel_filter = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels,
                                     fmin=fmin, fmax=fmax, htk=True, norm="slaney").astype(np.float32)
    out = []
    for c in chunks:
        spec = np.abs(librosa.stft(c, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                                    window="hann", center=True, pad_mode="reflect"))**2
        mel = mel_filter @ spec
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
        mn, mx = mel_db.min(), mel_db.max()
        mel_n = (mel_db - mn) / (mx - mn + 1e-8)
        # 3-channel (replicate)
        out.append(np.stack([mel_n, mel_n, mel_n]))
    return np.stack(out).astype(np.float32)


def nfnet_mel(chunks):
    """NFNet uses 128 mels @ 32k, fmin=50, fmax=14000."""
    mf = librosa.filters.mel(sr=SR, n_fft=2048, n_mels=128, fmin=50, fmax=14000,
                              htk=True, norm="slaney").astype(np.float32)
    out = []
    for c in chunks:
        spec = np.abs(librosa.stft(c, n_fft=2048, hop_length=512, win_length=2048,
                                    window="hann", center=True, pad_mode="reflect"))**2
        mel = mf @ spec
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
        mn, mx = mel_db.min(), mel_db.max()
        out.append(np.stack([(mel_db - mn) / (mx - mn + 1e-8)]*3))
    return np.stack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="kaggle_model")
    ap.add_argument("--soundscape_dir", default="data/train_soundscapes")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--out", default="pseudo_labels_v137/raw_predictions.csv")
    ap.add_argument("--power", type=float, default=2.0, help="PowerTransform exponent")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N files (debug)")
    args = ap.parse_args()

    tax = pd.read_csv(args.taxonomy)
    label_cols = tax["primary_label"].astype(str).tolist()
    N_CLASSES = len(label_cols)

    sc_dir = Path(args.soundscape_dir)
    files = sorted(sc_dir.glob("*.ogg"))
    if args.limit > 0:
        files = files[:args.limit]
    print(f"Files: {len(files)}")

    # Setup models — Perch is skipped (its ONNX is in a separate Kaggle dataset, not local;
    # also Perch outputs 14K classes needing nontrivial mapping). Using SED+NFNet+V2S.
    print("Loading models (SED + NFNet + V2S)...")
    sed_sess, sed_in = setup_sed(args.model_dir)
    nfnet_sess, nfnet_in = setup_nfnet(args.model_dir)
    v2s_sess, v2s_in = setup_v2s(args.model_dir, fold=1)
    print("Skipping Perch — using SED+NFNet+V2S ensemble (3 of V137's 4 models)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for fi, fp in enumerate(files):
        try:
            audio = read_60s(fp)
            chunks = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)

            # SED + V2S share mel
            mel_224 = fast_mel(chunks)
            sed_p = sed_sess.run(None, {sed_in: mel_224})[0].astype(np.float32)
            v2s_p = v2s_sess.run(None, {v2s_in: mel_224})[0].astype(np.float32)

            # NFNet has different mel
            mel_128 = nfnet_mel(chunks)
            nfnet_logits = nfnet_sess.run(None, {nfnet_in: mel_128})[0].astype(np.float32)
            nfnet_p = 1.0 / (1.0 + np.exp(-np.clip(nfnet_logits, -30, 30)))

            # Weighted average (V137 weights: 0.30 SED + 0.20 NFNet + 0.20 V2S, normalized = 0.43, 0.29, 0.29)
            ens = 0.43 * sed_p + 0.29 * nfnet_p + 0.29 * v2s_p

            # PowerTransform sharpen
            ens_sharp = np.power(ens, args.power)
            # Re-normalize per chunk so sum stays in reasonable range (don't normalize to 1 since multi-label)
            # Just clip to [0,1]
            ens_sharp = np.clip(ens_sharp, 0, 1)

            # Save per-chunk row with end_time = (i+1)*5
            for ci in range(N_WINDOWS):
                row = {"file": fp.name, "end_time": (ci+1) * 5.0}
                for li, lbl in enumerate(label_cols):
                    row[lbl] = float(ens_sharp[ci, li])
                rows.append(row)

            if (fi + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (fi + 1) / elapsed
                eta = (len(files) - fi - 1) / rate
                print(f"  {fi+1}/{len(files)} ({rate:.1f}/s, ETA {eta/60:.1f}min)")
        except Exception as e:
            print(f"  skip {fp.name}: {e}")

    print(f"\nTotal: {time.time()-t0:.0f}s, {len(rows)} pseudo rows")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Stats
    pp = df.iloc[:, 2:].values
    print(f"Pseudo stats: shape={pp.shape}, max={pp.max():.4f}, mean={pp.mean():.4f}")
    print(f"Pseudo per-row max histogram:")
    rowmax = pp.max(axis=1)
    for thr in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"  >{thr}: {(rowmax > thr).sum()} ({(rowmax > thr).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
