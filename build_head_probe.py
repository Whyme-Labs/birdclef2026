"""Build head_weights_train_audio.npz for V226 head_probe.

Runs Perch v2 ONNX on data/train_audio (35k focal recordings) to extract
1536-d embeddings per file. Then fits per-class linear classifier (W, b)
with trained_mask indicating which classes had enough positives.

Output: head_weights_train_audio.npz
  W: (n_classes, 1536)  float32 — per-class weights
  b: (n_classes,)       float32 — per-class biases
  trained_mask: (n_classes,) bool — True if class had >=5 positives
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import onnxruntime as ort
from sklearn.linear_model import LogisticRegression

SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC


def load_5s(path: Path, sr: int = SR, target_n: int = WINDOW_SAMPLES) -> np.ndarray:
    """Load a 5-second audio clip from focal recording (center crop)."""
    try:
        wav, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if file_sr != sr:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=file_sr, target_sr=sr)
        n = len(wav)
        if n < target_n:
            # Pad short clips to 5s by repeating
            reps = (target_n // n) + 1
            wav = np.tile(wav, reps)[:target_n]
        elif n > target_n:
            # Center crop
            start = (n - target_n) // 2
            wav = wav[start:start + target_n]
        return wav.astype(np.float32)
    except Exception as e:
        print(f"  ERR loading {path}: {e}")
        return np.zeros(target_n, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perch_onnx", default="local_perch/perch_v2.onnx")
    ap.add_argument("--train_audio", default="data/train_audio")
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--taxonomy_csv", default="data/taxonomy.csv")
    ap.add_argument("--emb_out", default="train_audio_perch_emb.npz")
    ap.add_argument("--head_out", default="head_weights_train_audio.npz")
    ap.add_argument("--max_per_class", type=int, default=200,
                    help="Cap samples per class for speed (random subsample)")
    ap.add_argument("--min_pos", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    # Load Perch ONNX
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(args.perch_onnx, opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print(f"Perch loaded: in={in_name}, outs={out_names}")
    # Identify embedding output (typically the 1536-d one)

    # Load train.csv + taxonomy
    df = pd.read_csv(args.train_csv)
    df["primary_label"] = df["primary_label"].astype(str)
    tax = pd.read_csv(args.taxonomy_csv)
    label_cols = sorted(tax["primary_label"].astype(str).tolist())
    l2i = {c: i for i, c in enumerate(label_cols)}
    n_classes = len(label_cols)
    print(f"  {len(df)} focal recordings, {n_classes} classes")

    # Subsample per class to cap total samples for speed
    rng = np.random.default_rng(42)
    keep_idx = []
    for cls, group in df.groupby("primary_label"):
        idxs = group.index.tolist()
        if len(idxs) > args.max_per_class:
            idxs = rng.choice(idxs, args.max_per_class, replace=False).tolist()
        keep_idx.extend(idxs)
    df_sub = df.iloc[keep_idx].reset_index(drop=True)
    print(f"  Subsampled to {len(df_sub)} files (max {args.max_per_class}/class)")

    # Check cache
    emb_path = Path(args.emb_out)
    if emb_path.exists():
        print(f"Loading cached embeddings from {emb_path}")
        d = np.load(emb_path)
        emb_all, lbl_all = d["emb"], d["labels"]
    else:
        emb_all = np.zeros((len(df_sub), 1536), dtype=np.float32)
        lbl_all = np.zeros((len(df_sub), n_classes), dtype=np.float32)
        # Multi-hot labels: primary_label + secondary_labels
        for i, row in df_sub.iterrows():
            primary = row["primary_label"]
            if primary in l2i:
                lbl_all[i, l2i[primary]] = 1.0
            # secondary_labels often a list-like string
            sec = row.get("secondary_labels", "")
            try:
                if isinstance(sec, str) and sec.startswith("["):
                    sec_list = eval(sec)
                    for s in sec_list:
                        s = str(s).strip()
                        if s in l2i:
                            lbl_all[i, l2i[s]] = 1.0
            except Exception:
                pass

        # Run Perch inference in batches
        n = len(df_sub)
        bs = args.batch_size
        for bs_i in range(0, n, bs):
            batch_files = df_sub.iloc[bs_i:bs_i+bs]
            batch_x = np.stack([load_5s(Path(args.train_audio) / fn) for fn in batch_files["filename"]])
            outs = sess.run(None, {in_name: batch_x})
            # Find which output is the 1536-d embedding (typically last or named "embedding")
            emb_out = None
            for o in outs:
                if o.ndim == 2 and o.shape[1] == 1536:
                    emb_out = o; break
            if emb_out is None:
                raise RuntimeError(f"No 1536-d output found. Outputs: {[o.shape for o in outs]}")
            emb_all[bs_i:bs_i+len(batch_files)] = emb_out
            if (bs_i // bs) % 50 == 0:
                elapsed = time.time() - t0
                pct = (bs_i + bs) / n * 100
                eta = elapsed / (bs_i + bs + 1) * (n - bs_i - bs)
                print(f"  {bs_i+bs}/{n} ({pct:.1f}%) — {elapsed:.0f}s elapsed, ~{eta:.0f}s ETA", flush=True)

        np.savez_compressed(emb_path, emb=emb_all, labels=lbl_all)
        print(f"  Saved embeddings to {emb_path} ({emb_path.stat().st_size/1e6:.1f} MB)")

    print(f"Perch pass: {time.time()-t0:.0f}s")
    print(f"  emb_all: {emb_all.shape}  lbl_all: {lbl_all.shape}")

    # Fit per-class linear classifier
    t1 = time.time()
    W = np.zeros((n_classes, 1536), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    trained_mask = np.zeros(n_classes, dtype=bool)
    pos_counts = lbl_all.sum(axis=0).astype(int)
    print(f"\nFitting linear heads (min_pos={args.min_pos})...")
    print(f"  Classes with ≥{args.min_pos} positives: {(pos_counts >= args.min_pos).sum()}/{n_classes}")

    # L2-normalize embeddings (helps LR conv)
    emb_norm = emb_all / (np.linalg.norm(emb_all, axis=1, keepdims=True) + 1e-8)

    for c in range(n_classes):
        y = lbl_all[:, c]
        n_pos = int(y.sum())
        if n_pos < args.min_pos: continue
        n_neg = (y == 0).sum()
        if n_neg < 50: continue  # need negatives too
        try:
            # Class-balanced weights to handle imbalance
            clf = LogisticRegression(
                max_iter=200, C=1.0, solver="liblinear",
                class_weight="balanced",
            )
            clf.fit(emb_norm, y)
            W[c] = clf.coef_[0].astype(np.float32)
            b[c] = clf.intercept_[0].astype(np.float32)
            trained_mask[c] = True
        except Exception as e:
            print(f"  fit failed for class {c} ({label_cols[c]}): {e}")

    print(f"  Fit {trained_mask.sum()}/{n_classes} classes in {time.time()-t1:.0f}s")

    # Save head weights
    np.savez_compressed(args.head_out,
                        W=W.astype(np.float32),
                        b=b.astype(np.float32),
                        trained_mask=trained_mask)
    print(f"Saved head weights to {args.head_out}")
    print(f"  Sample W norm range: [{np.linalg.norm(W[trained_mask], axis=1).min():.3f}, "
          f"{np.linalg.norm(W[trained_mask], axis=1).max():.3f}]")


if __name__ == "__main__":
    main()
