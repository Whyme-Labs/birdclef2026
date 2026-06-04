"""Diagnostic: per-class AUC of the public SED 5-fold on labeled soundscapes.

Identifies which of the 234 classes the public pipeline is weak on — the
candidates for the 0.949 -> 0.962 gap (hengck23: ~6 classes cost ~0.015 LB).
"""
import os
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import onnxruntime as ort
from sklearn.metrics import roc_auc_score

DATA = "/home/soh/birdclef-2026/data"
SED_DIR = "/home/soh/birdclef-2026/public_sed"

SR = 32000
N_MELS = 256
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 16000

# ── Load taxonomy
tax = pd.read_csv(f"{DATA}/taxonomy.csv")
tax["primary_label"] = tax["primary_label"].astype(str)
PRIMARY = sorted(tax["primary_label"].tolist())
l2i = {l: i for i, l in enumerate(PRIMARY)}
NC = len(PRIMARY)
cls_to_taxon = dict(zip(tax["primary_label"], tax["class_name"]))
print(f"Classes: {NC}")

# ── Load labeled soundscapes
df = pd.read_csv(f"{DATA}/train_soundscapes_labels.csv")
df["filename"] = df["filename"].astype(str)
print(f"Labeled windows: {len(df)} from {df['filename'].nunique()} files")

# ── Build truth matrix
Y = np.zeros((len(df), NC), dtype=np.float32)
for i, row in df.iterrows():
    labels = str(row["primary_label"]).split(";")
    for lab in labels:
        lab = lab.strip()
        if lab in l2i:
            Y[i, l2i[lab]] = 1.0
print(f"Truth matrix: {Y.shape}, positives per window mean: {Y.sum(1).mean():.2f}")
print(f"Classes with >=1 positive: {(Y.sum(0) > 0).sum()}/{NC}")

# ── Load SED 5-fold
sessions = []
for f in range(5):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(f"{SED_DIR}/sed_fold{f}.onnx", sess_options=so,
                              providers=["CPUExecutionProvider"])
    sessions.append(s)
inp_name = sessions[0].get_inputs()[0].name
out_names = [o.name for o in sessions[0].get_outputs()]
clip_out = next((n for n in out_names if "clip" in n.lower()), out_names[0])
frame_out = next((n for n in out_names if "frame" in n.lower()), None)
print(f"SED loaded. input={inp_name}, clip={clip_out}, frame={frame_out}")


def parse_time(t):
    parts = str(t).split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def make_mel(wav):
    s = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    s = librosa.power_to_db(s, top_db=80)
    s = (s - s.mean()) / (s.std() + 1e-6)
    if s.shape[-1] < 313:
        s = np.pad(s, ((0, 0), (0, 313 - s.shape[-1])))
    else:
        s = s[:, :313]
    return s[None, None, :, :].astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ── Run SED on each labeled window
preds = np.zeros((len(df), NC), dtype=np.float32)
audio_cache = {}
t0 = time.time()
for i, row in df.iterrows():
    fn = row["filename"]
    if fn not in audio_cache:
        path = f"{DATA}/train_soundscapes/{fn}"
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            audio_cache[fn] = wav
        except Exception as e:
            print(f"  read error {fn}: {e}")
            audio_cache[fn] = None
    wav = audio_cache[fn]
    if wav is None:
        continue
    st = int(parse_time(row["start"]) * SR)
    en = int(parse_time(row["end"]) * SR)
    chunk = wav[st:en]
    if len(chunk) < SR * 5:
        chunk = np.pad(chunk, (0, SR * 5 - len(chunk)))
    else:
        chunk = chunk[:SR * 5]
    mel = make_mel(chunk)
    p = np.zeros(NC, dtype=np.float32)
    for sess in sessions:
        outs = sess.run(None, {inp_name: mel})
        clip = outs[0][0]  # (234,)
        if frame_out and len(outs) > 1:
            frame = outs[1][0]  # (T, 234) or similar
            frame_max = frame.max(axis=0) if frame.ndim == 2 else frame
            p += (0.5 * sigmoid(clip) + 0.5 * sigmoid(frame_max)) / len(sessions)
        else:
            p += sigmoid(clip) / len(sessions)
    preds[i] = p
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(df)} t={time.time()-t0:.0f}s", flush=True)

print(f"SED inference done in {time.time()-t0:.0f}s")

# ── Per-class AUC
results = []
for ci in range(NC):
    pos = int(Y[:, ci].sum())
    if pos == 0 or pos == len(Y):
        continue
    try:
        auc = roc_auc_score(Y[:, ci], preds[:, ci])
    except Exception:
        continue
    results.append({
        "class": PRIMARY[ci],
        "taxon": cls_to_taxon.get(PRIMARY[ci], "?"),
        "n_pos": pos,
        "auc": auc,
    })

res_df = pd.DataFrame(results).sort_values("auc")
macro = res_df["auc"].mean()
print(f"\n=== SED-only macro AUC on labeled soundscapes: {macro:.4f} ===")
print(f"Classes evaluated: {len(res_df)}")
print(f"\n--- 25 WORST classes (SED weak spots) ---")
print(res_df.head(25).to_string(index=False))
print(f"\n--- AUC distribution by taxon ---")
print(res_df.groupby("taxon")["auc"].agg(["count", "mean", "min"]).to_string())

res_df.to_csv("/home/soh/birdclef-2026/diagnostic_perclass_auc.csv", index=False)
print(f"\nSaved full table to diagnostic_perclass_auc.csv")
