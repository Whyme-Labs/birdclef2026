"""Validate BioLingual zero-shot signal on labeled soundscape chunks."""
import pandas as pd
import numpy as np
import torch
import soundfile as sf
import librosa
from pathlib import Path
from transformers import ClapModel, ClapProcessor
from sklearn.metrics import roc_auc_score

# Load data
labels_df = pd.read_csv("data/train_soundscapes_labels.csv")
tax = pd.read_csv("data/taxonomy.csv")
species_to_idx = {s: i for i, s in enumerate(tax["primary_label"].astype(str).tolist())}
N_SPECIES = len(tax)
print(f"Total labels: {len(labels_df)}, species: {N_SPECIES}")

# Convert primary_label (semicolon-sep) to multi-hot
def to_multihot(s):
    if pd.isna(s) or not s: return np.zeros(N_SPECIES, dtype=np.float32)
    out = np.zeros(N_SPECIES, dtype=np.float32)
    for sp in str(s).split(";"):
        sp = sp.strip()
        if sp in species_to_idx: out[species_to_idx[sp]] = 1.0
    return out

# Sample 300 chunks for quick validation, balanced across files
np.random.seed(42)
sample_n = min(300, len(labels_df))
sampled = labels_df.sample(n=sample_n, random_state=42).reset_index(drop=True)
print(f"Sampled: {sample_n}")

# Build labels
Y = np.stack([to_multihot(s) for s in sampled["primary_label"]])
print(f"Labels: {Y.shape}, total positives: {Y.sum():.0f}, mean per chunk: {Y.sum(1).mean():.2f}")

# Load BioLingual
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ClapModel.from_pretrained("davidrrobinson/BioLingual").to(device)
processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")
model.eval()

# Load text embeddings
text_embs = np.load("kaggle_model/biolingual_text_embs.npy")  # (234, 512)
text_embs_t = torch.from_numpy(text_embs).to(device)
print(f"Text embeddings loaded: {text_embs.shape}")

# Process audio chunks
SR = 48000
CHUNK_LEN_S = 5.0  # 5-second chunks (BirdCLEF inference granularity)
audio_root = Path("data/train_soundscapes")

def load_chunk(filename, start_str, end_str):
    """Load 5s chunk from filename at start time."""
    path = audio_root / filename
    if not path.exists(): return None
    # Parse HH:MM:SS
    h, m, s = start_str.split(":")
    start_s = int(h)*3600 + int(m)*60 + int(s)
    audio, sr = sf.read(str(path), start=start_s*sr if False else None)
    # safer: librosa with offset
    audio, sr = librosa.load(str(path), sr=SR, offset=start_s, duration=CHUNK_LEN_S, mono=True)
    return audio

scores = np.zeros((sample_n, N_SPECIES), dtype=np.float32)
batch_audios = []
batch_indices = []
BATCH = 16

print("Encoding audio...")
import time
t0 = time.time()
for i in range(sample_n):
    row = sampled.iloc[i]
    audio = load_chunk(row["filename"], row["start"], row["end"])
    if audio is None or len(audio) < SR:
        continue
    if len(audio) < int(SR * CHUNK_LEN_S):
        audio = np.pad(audio, (0, int(SR*CHUNK_LEN_S) - len(audio)))
    batch_audios.append(audio)
    batch_indices.append(i)
    
    if len(batch_audios) >= BATCH or i == sample_n - 1:
        if not batch_audios: continue
        with torch.no_grad():
            inputs = processor(audios=batch_audios, sampling_rate=SR, return_tensors="pt").to(device)
            audio_feats = model.get_audio_features(**inputs)
            audio_feats = torch.nn.functional.normalize(audio_feats, dim=-1)
            sim = (audio_feats @ text_embs_t.T).cpu().numpy()
        for j, idx in enumerate(batch_indices):
            scores[idx] = sim[j]
        batch_audios = []
        batch_indices = []
    if i % 50 == 0 and i > 0:
        elapsed = time.time() - t0
        rate = i / elapsed
        eta = (sample_n - i) / rate
        print(f"  {i}/{sample_n} done, rate {rate:.1f}/s, ETA {eta:.0f}s")

print(f"Total time: {time.time()-t0:.1f}s")

# Compute AUC per species (only species with at least 5 positives in sample)
pos_per_species = Y.sum(axis=0)
valid = pos_per_species >= 5
print(f"\nSpecies with ≥5 positives in sample: {valid.sum()}")
if valid.sum() > 0:
    aucs = []
    for c in np.where(valid)[0]:
        try:
            auc = roc_auc_score(Y[:, c], scores[:, c])
            aucs.append(auc)
        except ValueError:
            pass
    aucs = np.array(aucs)
    print(f"Per-species AUC: mean={aucs.mean():.4f}, median={np.median(aucs):.4f}")
    print(f"  Best 5: {sorted(aucs, reverse=True)[:5]}")
    print(f"  Worst 5: {sorted(aucs)[:5]}")
    
# Macro AUC over chunks (any species ≥5 pos)
score_subset = scores[:, valid]
y_subset = Y[:, valid]
try:
    macro_auc = roc_auc_score(y_subset, score_subset, average="macro")
    print(f"Macro AUC: {macro_auc:.4f}")
except ValueError as e:
    print(f"Macro AUC failed: {e}")

# Save scores for later analysis
np.save("/tmp/biolingual_val_scores.npy", scores)
np.save("/tmp/biolingual_val_labels.npy", Y)
print("\nSaved to /tmp/biolingual_val_*.npy")
