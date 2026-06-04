"""Smoke test the V174 ONNX + probe pipeline against true labels."""
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import onnxruntime as ort
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Setup
SR = 48000
NFFT, HOP, NMELS = 1024, 480, 64
FMIN, FMAX = 50.0, 14000.0
MAX_SAMPLES = 480_000
NB_FRAMES = 1001

_window = np.hanning(NFFT).astype(np.float32)
_mel_filters = librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=NMELS,
                                    fmin=FMIN, fmax=FMAX, htk=False, norm="slaney").astype(np.float32)

def make_mel(wav):
    if wav.shape[0] < MAX_SAMPLES:
        n_repeat = MAX_SAMPLES // wav.shape[0]
        wav_padded = np.tile(wav, n_repeat)
        if wav_padded.shape[0] < MAX_SAMPLES:
            wav_padded = np.pad(wav_padded, (0, MAX_SAMPLES - wav_padded.shape[0]))
    else:
        wav_padded = wav[:MAX_SAMPLES]
    stft = librosa.stft(wav_padded.astype(np.float32), n_fft=NFFT, hop_length=HOP,
                        win_length=NFFT, window="hann", center=True, pad_mode="reflect")
    power = np.abs(stft) ** 2
    mel = _mel_filters @ power
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10)).T
    log_mel = log_mel[:NB_FRAMES, :]
    if log_mel.shape[0] < NB_FRAMES:
        log_mel = np.pad(log_mel, ((0, NB_FRAMES - log_mel.shape[0]), (0, 0)))
    return log_mel

# Load probes
data = np.load("kaggle_model/biolingual_probes_v174.npz", allow_pickle=True)
W = data["weights"].astype(np.float32)
b = data["biases"].astype(np.float32)
mask = data["mask"].astype(bool)
print(f"Probes: W={W.shape}, mask active={mask.sum()}/{len(mask)}")

# Load ONNX
sess = ort.InferenceSession("kaggle_model/biolingual_audio.onnx", providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name

# Load labels
sc = pd.read_csv("data/train_soundscapes_labels.csv")
tax = pd.read_csv("data/taxonomy.csv")
label_cols = tax["primary_label"].astype(str).tolist()
l2i = {c: i for i, c in enumerate(label_cols)}

# Sample 100 chunks
np.random.seed(0)
sampled = sc.sample(n=200, random_state=0).reset_index(drop=True)

audio_dir = Path("data/train_soundscapes")
all_scores = []
all_labels = []
for i in range(len(sampled)):
    row = sampled.iloc[i]
    fp = audio_dir / row["filename"]
    if not fp.exists(): continue
    h, m, s = row["start"].split(":")
    start_s = int(h)*3600 + int(m)*60 + int(s)
    audio, sr = librosa.load(str(fp), sr=SR, offset=start_s, duration=5.0, mono=True)
    if len(audio) < SR: continue
    audio = audio.astype(np.float32)
    if len(audio) < 5*SR: audio = np.pad(audio, (0, 5*SR - len(audio)))

    mel = make_mel(audio)[None, None, :, :]
    emb = sess.run(None, {in_name: mel})[0][0]  # (512,)
    logits = emb @ W.T + b
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

    Y = np.zeros(len(label_cols), dtype=np.float32)
    if pd.notna(row["primary_label"]):
        for lbl in str(row["primary_label"]).split(";"):
            lbl = lbl.strip()
            if lbl in l2i: Y[l2i[lbl]] = 1.0
    all_scores.append(probs)
    all_labels.append(Y)

S = np.array(all_scores)
L = np.array(all_labels)
print(f"Scores: {S.shape}, Labels: {L.shape}, total positives: {L.sum():.0f}")

# Per-class AUC for active probes with ≥5 positives
pos = L.sum(axis=0)
valid = mask & (pos >= 5)
print(f"Valid for AUC: {valid.sum()} species")
aucs = []
for c in np.where(valid)[0]:
    try:
        a = roc_auc_score(L[:, c], S[:, c])
        aucs.append(a)
    except: pass
aucs = np.array(aucs)
print(f"Per-species AUC (probed species): mean={aucs.mean():.4f}, median={np.median(aucs):.4f}")
print(f"Best 5: {sorted(aucs, reverse=True)[:5]}")
print(f"Worst 5: {sorted(aucs)[:5]}")

# Macro AUC
try:
    macro = roc_auc_score(L[:, valid], S[:, valid], average="macro")
    print(f"Macro AUC: {macro:.4f}")
except Exception as e:
    print(f"Macro AUC failed: {e}")
