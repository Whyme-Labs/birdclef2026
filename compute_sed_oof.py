"""
Compute SED OOF predictions on the 59 fully-labeled soundscape files.
These predictions are used to optimize ensemble weights with Perch OOF.
"""
import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
import librosa
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import time

DATA_ROOT = Path("/home/soh/birdclef-2026/data")
MODEL_DIR = Path("/home/soh/birdclef-2026/pretrained")
OUT_DIR = Path("/home/soh/birdclef-2026/kaggle_model")

# Load metadata
sample_sub = pd.read_csv(DATA_ROOT / "sample_submission.csv")
PRIMARY_LABELS = list(sample_sub.columns[1:])
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

sc = pd.read_csv(DATA_ROOT / "train_soundscapes_labels.csv")

def parse_labels(x):
    if pd.isna(x): return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_labels(x)))

sc_clean = (
    sc.drop_duplicates()
    .groupby(["filename", "start", "end"])["primary_label"]
    .apply(union_labels)
    .reset_index(name="label_list")
)
sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
sc_clean["row_id"] = (
    sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)
)

N_WINDOWS = 12
wpf = sc_clean.groupby("filename").size()
full_files = sorted(wpf[wpf == N_WINDOWS].index.tolist())
sc_full = sc_clean[sc_clean["filename"].isin(full_files)].sort_values(["filename", "end_sec"]).reset_index(drop=True)

print(f"Fully-labeled files: {len(full_files)}")
print(f"Windows: {len(sc_full)}")

# Build ground truth
Y = np.zeros((len(sc_full), N_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_full["label_list"]):
    for lbl in labels:
        if lbl in label_to_idx:
            Y[i, label_to_idx[lbl]] = 1

# SED config (must match notebook exactly)
SR = 32000
CHUNK_SAMPLES = SR * 5
FILE_SAMPLES = 60 * SR
N_FFT = 2048
HOP = 512
N_MELS = 224
FMIN = 0
FMAX = 16000
TOP_DB = 80.0

mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, htk=True, norm="slaney")

def audio_to_mel_batch(chunks):
    batch = []
    for chunk in chunks:
        S = np.abs(librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP, window="hann")) ** 2
        mel = mel_basis @ S
        mel_db = librosa.power_to_db(mel, top_db=TOP_DB)
        mn, mx = mel_db.min(), mel_db.max()
        mel_db = (mel_db - mn) / (mx - mn + 1e-7)
        batch.append(mel_db)
    mel_arr = np.stack(batch)[:, np.newaxis]
    mel_arr = np.repeat(mel_arr, 3, axis=1)
    return mel_arr.astype(np.float32)

def sed_postprocess(probs, sharpen_exp=1.5, smooth_sigma=0.7, file_max_weight=0.05):
    if probs.shape[0] <= 1:
        return probs
    file_max = probs.max(axis=0, keepdims=True)
    probs = probs + file_max_weight * file_max
    sharpened = np.power(probs, sharpen_exp)
    smoothed = gaussian_filter1d(sharpened, sigma=smooth_sigma, axis=0)
    return np.power(np.maximum(smoothed, 1e-10), 1.0 / sharpen_exp)

# Load ONNX models
BLEND = {"LB862": 0.2, "LB872": 0.8}
sessions = {}
for name in BLEND:
    path = MODEL_DIR / f"{name}.onnx"
    sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"Loaded {name}.onnx")

def predict_chunks(chunks_audio):
    mel = audio_to_mel_batch(chunks_audio)
    probs = np.zeros((len(chunks_audio), N_CLASSES), dtype=np.float64)
    total_w = 0.0
    for name, sess in sessions.items():
        w = BLEND[name]
        out = sess.run(None, {"mel": mel})[0]
        probs += w * out
        total_w += w
    probs /= total_w
    return probs

# Run SED on all fully-labeled files
t0 = time.time()
all_preds = np.zeros((len(sc_full), N_CLASSES), dtype=np.float64)
row_idx = 0

for fi, fname in enumerate(full_files):
    fpath = DATA_ROOT / "train_soundscapes" / fname
    y, sr = sf.read(str(fpath), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    y = y[:FILE_SAMPLES]

    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    chunks = y.reshape(N_WINDOWS, CHUNK_SAMPLES)
    probs = predict_chunks(chunks)
    probs = sed_postprocess(probs)

    all_preds[row_idx:row_idx + N_WINDOWS] = probs
    row_idx += N_WINDOWS

    if (fi + 1) % 10 == 0:
        print(f"  {fi+1}/{len(full_files)} files ({time.time()-t0:.0f}s)")

print(f"SED OOF done: {all_preds.shape} in {time.time()-t0:.0f}s")

# Compute per-class AUC
from sklearn.metrics import roc_auc_score

macro_aucs = []
per_class_auc = {}
for i in range(N_CLASSES):
    if Y[:, i].sum() > 0 and Y[:, i].sum() < len(Y):
        auc = roc_auc_score(Y[:, i], all_preds[:, i])
        macro_aucs.append(auc)
        per_class_auc[PRIMARY_LABELS[i]] = auc

print(f"\nSED OOF macro-AUC: {np.mean(macro_aucs):.4f} ({len(macro_aucs)} evaluable classes)")

# Save
np.savez_compressed(
    OUT_DIR / "sed_oof_preds.npz",
    predictions=all_preds.astype(np.float32),
    labels=Y,
    row_ids=sc_full["row_id"].values,
)
print(f"Saved to {OUT_DIR / 'sed_oof_preds.npz'}")

# Show worst classes
sorted_auc = sorted(per_class_auc.items(), key=lambda x: x[1])
print("\nWorst 20 classes by SED AUC:")
for name, auc in sorted_auc[:20]:
    print(f"  {name}: {auc:.3f} (n_pos={Y[:, label_to_idx[name]].sum()})")

print("\nBest 10 classes:")
for name, auc in sorted_auc[-10:]:
    print(f"  {name}: {auc:.3f} (n_pos={Y[:, label_to_idx[name]].sum()})")
