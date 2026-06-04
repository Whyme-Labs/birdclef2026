"""V162: train per-class linear probes in AudioMAE space using labeled soundscape data.

This is the principled fix for V160's failure: V160 built templates from focal
recordings (close-mic'd, clean) which don't transfer to soundscape inference.
V162 trains the probes on actual labeled SOUNDSCAPE chunks — same recipe as
V137's Perch probes, just in AudioMAE embedding space.

Outputs:
  • kaggle_model/audiomae_probes_v162.npz — per-class linear weights and biases
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
import librosa
import soundfile as sf
from sklearn.linear_model import LogisticRegression


def audiomae_mel(chunks):
    _pad = 1024
    _am_window = np.hanning(2048).astype(np.float32)
    _am_mel = librosa.filters.mel(sr=32000, n_fft=2048, n_mels=128,
                                  fmin=20, fmax=16000, htk=True, norm="slaney").astype(np.float32)
    batch = []
    for chunk in chunks:
        peak = np.max(np.abs(chunk))
        if peak > 0:
            chunk = chunk / peak
        cp = np.pad(chunk, (_pad, _pad), mode="reflect")
        nf = 1 + (len(cp) - 2048) // 500
        frames = np.lib.stride_tricks.as_strided(
            cp, (nf, 2048), (cp.strides[0]*500, cp.strides[0])).copy()
        spec = np.abs(np.fft.rfft(frames * _am_window, axis=1))**2
        mel = _am_mel @ spec.T
        mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
        mn, mx = mel_db.min(), mel_db.max()
        mel_db = (mel_db - mn) / (mx - mn + 1e-7)
        mel_t = mel_db.T
        target_T = (mel_t.shape[0] // 16) * 16
        mel_t = mel_t[:target_T, :]
        batch.append(mel_t.astype(np.float32))
    return np.stack(batch)[:, np.newaxis, :, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audiomae_onnx", default="kaggle_model/audiomae_emb_v160.onnx")
    ap.add_argument("--soundscape_dir", default="data/train_soundscapes")
    ap.add_argument("--soundscape_csv", default="data/train_soundscapes_labels.csv")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--out", default="kaggle_model/audiomae_probes_v162.npz")
    ap.add_argument("--min_pos", type=int, default=3, help="min positives per class to train probe")
    ap.add_argument("--C", type=float, default=1.0, help="LR regularization")
    args = ap.parse_args()

    # Setup labels
    tax = pd.read_csv(args.taxonomy)
    label_cols = sorted(tax["primary_label"].astype(str).tolist())
    l2i = {c: i for i, c in enumerate(label_cols)}
    print(f"{len(label_cols)} classes")

    # Soundscape labels: parse into per-(file, end_sec) labels
    sc = pd.read_csv(args.soundscape_csv)
    print(f"  raw labels rows: {len(sc)}")

    # Build per-chunk label matrix — primary_label is SEMICOLON-SEPARATED multi-label
    sc["primary_label"] = sc["primary_label"].astype(str)
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    def parse_lbl(x):
        if pd.isna(x) or x == "nan": return set()
        return set(t.strip() for t in str(x).split(";") if t.strip())
    def union_sets(series):
        out = set()
        for s in series:
            out |= s
        return out
    sc["label_set"] = sc["primary_label"].apply(parse_lbl)
    grouped = sc.groupby(["filename", "end_sec"])["label_set"].apply(union_sets).reset_index()
    grouped = grouped.rename(columns={"label_set": "primary_label"})
    print(f"  unique chunks: {len(grouped)}")

    # Filter to existing audio files
    audio_dir = Path(args.soundscape_dir)
    existing_files = set(p.name for p in audio_dir.glob("*.ogg"))
    grouped = grouped[grouped["filename"].isin(existing_files)].reset_index(drop=True)
    print(f"  chunks with audio: {len(grouped)}")

    # AudioMAE backbone
    print(f"\nLoading AudioMAE: {args.audiomae_onnx}")
    sess = ort.InferenceSession(args.audiomae_onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    # For each chunk: read audio, extract embedding
    n_chunks = len(grouped)
    X = np.zeros((n_chunks, 768), dtype=np.float32)
    Y = np.zeros((n_chunks, len(label_cols)), dtype=np.uint8)

    # Group by filename for efficient audio reading
    print("\nExtracting embeddings…")
    t0 = time.time()
    by_file = grouped.groupby("filename")
    chunk_index = 0
    for fi, (fn, group) in enumerate(by_file):
        fp = audio_dir / fn
        try:
            wav, sr = sf.read(str(fp))
        except Exception as e:
            print(f"  skip {fn}: {e}")
            chunk_index += len(group)
            continue
        if sr != 32000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=32000)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # Extract chunks based on end_sec
        chunk_audios = []
        idx_list = []
        for _, row in group.iterrows():
            end_sec = row["end_sec"]
            start = (end_sec - 5) * 32000
            end = end_sec * 32000
            if start < 0 or end > len(wav):
                # Pad / clip
                target = wav[max(0, start):min(len(wav), end)]
                if len(target) < 5*32000:
                    target = np.pad(target, (0, 5*32000 - len(target)))
                chunk_audios.append(target.astype(np.float32))
            else:
                chunk_audios.append(wav[start:end].astype(np.float32))
            idx_list.append(_)
        if not chunk_audios:
            continue
        mel = audiomae_mel(chunk_audios)
        emb = sess.run(None, {inp: mel})[0]  # (n_chunks, 768)

        for k, idx in enumerate(idx_list):
            row = group.iloc[k]
            slot = grouped.index.get_loc(group.index[k])
            X[slot] = emb[k]
            for lbl in row["primary_label"]:
                if lbl in l2i:
                    Y[slot, l2i[lbl]] = 1

        if (fi + 1) % 5 == 0:
            print(f"  {fi+1}/{len(by_file)} files — {time.time()-t0:.0f}s")

    print(f"Embeddings done in {time.time()-t0:.0f}s, X={X.shape} Y={Y.shape}")
    print(f"Y per-class positives: min={Y.sum(0).min()} median={int(np.median(Y.sum(0)))} max={Y.sum(0).max()}")
    print(f"  classes with ≥{args.min_pos} positives: {(Y.sum(0) >= args.min_pos).sum()}")

    # Train probes
    print("\nTraining probes…")
    weights = np.zeros((len(label_cols), 768), dtype=np.float32)
    biases = np.zeros(len(label_cols), dtype=np.float32)
    mask = np.zeros(len(label_cols), dtype=bool)

    pos_counts = Y.sum(axis=0)
    for ci in range(len(label_cols)):
        n_pos = int(pos_counts[ci])
        if n_pos < args.min_pos:
            continue
        y = Y[:, ci].astype(int)
        try:
            clf = LogisticRegression(C=args.C, max_iter=200, solver="liblinear", class_weight="balanced")
            clf.fit(X, y)
            weights[ci] = clf.coef_[0]
            biases[ci] = clf.intercept_[0]
            mask[ci] = True
        except Exception as e:
            print(f"  skip class {label_cols[ci]} ({n_pos} pos): {e}")

    print(f"Probes trained: {mask.sum()} / {len(label_cols)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        weights=weights, biases=biases, mask=mask,
                        label_cols=np.array(label_cols),
                        pos_counts=pos_counts)
    print(f"Saved: {args.out}")

    # V190 orthogonality: also save raw features for cross-probe analysis
    emb_out = "kaggle_model/audiomae_train_emb.npz"
    np.savez_compressed(emb_out, X=X, Y=Y, label_cols=np.array(label_cols))
    print(f"Saved features: {emb_out}")


if __name__ == "__main__":
    main()
