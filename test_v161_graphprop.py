"""Local validation: does AudioMAE-graph label propagation improve macro AUC on OOF SED preds?"""
import time
import numpy as np
import onnxruntime as ort
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.metrics import roc_auc_score


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


def macro_auc(labels, preds):
    """Per-class AUC, macro-averaged over classes that have at least one positive."""
    aucs = []
    for j in range(labels.shape[1]):
        if labels[:, j].sum() > 0 and labels[:, j].sum() < len(labels[:, j]):
            try:
                aucs.append(roc_auc_score(labels[:, j], preds[:, j]))
            except ValueError:
                pass
    return np.mean(aucs), len(aucs)


def graph_propagate(preds_3d, emb_3d, tau, alpha, K):
    """
    preds_3d: (n_files, 12, 234)
    emb_3d: (n_files, 12, 768) L2-normed
    Returns propagated preds same shape.
    """
    out = preds_3d.copy()
    for fi in range(preds_3d.shape[0]):
        e = emb_3d[fi]
        sim = e @ e.T
        logits = sim * tau
        logits = logits - logits.max(axis=1, keepdims=True)
        W = np.exp(logits)
        W = W / W.sum(axis=1, keepdims=True)
        p0 = preds_3d[fi].astype(np.float32)
        p = p0.copy()
        for _ in range(K):
            p = (1 - alpha) * p0 + alpha * (W @ p)
        out[fi] = p
    return out


def main():
    print("Loading OOF…")
    d = np.load("kaggle_model/sed_oof_preds.npz", allow_pickle=True)
    preds = d["predictions"]      # (708, 234)
    labels = d["labels"]          # (708, 234)
    row_ids = d["row_ids"]
    print(f"  preds={preds.shape} labels={labels.shape}")

    # Group by file
    files_to_idx = {}  # filename → list of (idx, end_sec)
    for i, rid in enumerate(row_ids):
        parts = rid.rsplit("_", 1)
        fn, end_sec = parts[0], int(parts[1])
        files_to_idx.setdefault(fn, []).append((i, end_sec))
    # Sort by end_sec to get correct order
    for fn in files_to_idx:
        files_to_idx[fn].sort(key=lambda x: x[1])
    print(f"  unique files: {len(files_to_idx)}")

    # Verify all files have 12 chunks
    chunk_counts = [len(v) for v in files_to_idx.values()]
    if min(chunk_counts) != 12 or max(chunk_counts) != 12:
        print(f"  WARN: chunk counts vary: min={min(chunk_counts)} max={max(chunk_counts)}")
        # Filter
        files_to_idx = {fn: v for fn, v in files_to_idx.items() if len(v) == 12}
        print(f"  using {len(files_to_idx)} files with 12 chunks")

    # Extract embeddings for these files
    print("\nLoading AudioMAE backbone…")
    sess = ort.InferenceSession("kaggle_model/audiomae_emb_v160.onnx",
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    print(f"  input: {inp}")

    audio_dir = Path("data/train_soundscapes")
    n_files = len(files_to_idx)
    n_chunks_total = n_files * 12
    emb_array = np.zeros((n_chunks_total, 768), dtype=np.float32)
    pred_array = np.zeros((n_chunks_total, 234), dtype=np.float32)
    label_array = np.zeros((n_chunks_total, 234), dtype=np.uint8)

    file_order = sorted(files_to_idx.keys())
    t0 = time.time()
    for fi, fn in enumerate(file_order):
        # Find audio file
        candidates = list(audio_dir.glob(f"{fn}*"))
        if not candidates:
            print(f"  MISSING: {fn}")
            continue
        fp = candidates[0]
        wav, sr = sf.read(str(fp))
        if sr != 32000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=32000)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # Get 12 chunks of 5s
        target_len = 60 * 32000
        if len(wav) >= target_len:
            wav = wav[:target_len]
        else:
            wav = np.pad(wav, (0, target_len - len(wav)))
        chunks = wav.reshape(12, 5 * 32000).astype(np.float32)

        mel_in = audiomae_mel(chunks)
        emb = sess.run(None, {inp: mel_in})[0]  # (12, 768)

        for ci, (orig_idx, end_sec) in enumerate(files_to_idx[fn]):
            slot = fi * 12 + ci
            emb_array[slot] = emb[ci]
            pred_array[slot] = preds[orig_idx]
            label_array[slot] = labels[orig_idx]
        if (fi + 1) % 10 == 0:
            print(f"  {fi+1}/{n_files} — {time.time()-t0:.0f}s")

    print(f"Embeddings done in {time.time()-t0:.0f}s")

    # Reshape to 3D
    preds_3d = pred_array.reshape(n_files, 12, 234)
    emb_3d = emb_array.reshape(n_files, 12, 768)
    labels_2d = label_array  # (n_chunks_total, 234)

    # Baseline AUC (no propagation)
    base_auc, n_classes = macro_auc(labels_2d, pred_array)
    print(f"\nBaseline macro AUC (no propagation): {base_auc:.4f} over {n_classes} classes")

    # Sweep hyperparams
    print(f"\n{'τ':>4} {'α':>5} {'K':>2} {'AUC':>8} {'Δ':>8}")
    print("-" * 35)
    for tau in [4.0, 8.0, 12.0, 16.0]:
        for alpha in [0.10, 0.20, 0.30, 0.50]:
            for K in [1, 2, 3]:
                propagated = graph_propagate(preds_3d, emb_3d, tau, alpha, K)
                propagated_2d = propagated.reshape(-1, 234)
                auc, _ = macro_auc(labels_2d, propagated_2d)
                delta = auc - base_auc
                marker = " *" if delta > 0 else ("  -" if delta < -0.001 else "")
                print(f"{tau:>4.1f} {alpha:>5.2f} {K:>2} {auc:>8.4f} {delta:>+8.4f}{marker}")


if __name__ == "__main__":
    main()
