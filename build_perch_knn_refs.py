"""
Build retrieval reference for V157: filter the cached Perch embeddings down to
just the 66 hand-labeled soundscape files (792 windows) and pair each window
with its multi-hot label vector.

Output: kaggle_model/perch_knn_refs.npz with arrays
  emb:    (n_ref, 1536) float32   — Perch embeddings (L2-normalized)
  labels: (n_ref, 234)  float32   — multi-hot labels
  meta:   (n_ref,)       int32    — global index for debugging (file_idx*12 + win_idx)
"""
import numpy as np
import pandas as pd
from pathlib import Path

EMB_NPZ = "kaggle_model/all_perch_arrays.npz"
META_PQ = "kaggle_model/all_perch_meta.parquet"
LABELS_CSV = "data/train_soundscapes_labels.csv"
TAXONOMY_CSV = "data/taxonomy.csv"
OUT = "kaggle_model/perch_knn_refs.npz"

tax = pd.read_csv(TAXONOMY_CSV)
label_cols = sorted(tax["primary_label"].astype(str).tolist())
l2i = {c: i for i, c in enumerate(label_cols)}

d = np.load(EMB_NPZ)
emb_full = d["emb_full"]
meta = pd.read_parquet(META_PQ)
files = list(meta["filename"].drop_duplicates())
file2idx = {f: i for i, f in enumerate(files)}
n_files = len(files)
n_windows = 12
emb_3d = emb_full.reshape(n_files, n_windows, -1)

labels_csv = pd.read_csv(LABELS_CSV)
labels_csv["primary_label"] = labels_csv["primary_label"].astype(str)
labels_csv["end_sec"] = pd.to_timedelta(labels_csv["end"]).dt.total_seconds().astype(int)

# Build (n_files, n_windows, 234) label tensor for the cached files
Y = np.zeros((n_files, n_windows, len(label_cols)), dtype=np.float32)
labeled_files = set()
for _, row in labels_csv.iterrows():
    fn = row["filename"]
    if fn not in file2idx:
        continue
    fi = file2idx[fn]
    wi = (row["end_sec"] // 5) - 1
    if 0 <= wi < n_windows:
        for lbl in str(row["primary_label"]).split(";"):
            lbl = lbl.strip()
            if lbl in l2i:
                Y[fi, wi, l2i[lbl]] = 1.0
        labeled_files.add(fn)

print(f"Labeled files among cached: {len(labeled_files)}")
labeled_idx = [file2idx[f] for f in sorted(labeled_files)]
emb_lab = emb_3d[labeled_idx].reshape(-1, emb_3d.shape[-1])
y_lab = Y[labeled_idx].reshape(-1, Y.shape[-1])
print(f"Reference set: emb {emb_lab.shape}, labels {y_lab.shape}, mean pos/win {y_lab.sum(axis=1).mean():.2f}")

# L2-normalize embeddings for cosine similarity
norm = np.linalg.norm(emb_lab, axis=1, keepdims=True)
emb_lab_n = (emb_lab / np.maximum(norm, 1e-9)).astype(np.float32)

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
meta_int = np.arange(emb_lab.shape[0], dtype=np.int32)
np.savez_compressed(OUT, emb=emb_lab_n, labels=y_lab.astype(np.float32), meta=meta_int)
print(f"Wrote {OUT}: {emb_lab_n.shape} embeddings, {y_lab.shape} labels")
