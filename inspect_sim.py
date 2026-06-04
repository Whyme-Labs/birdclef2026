"""Inspect AudioMAE chunk-similarity within a soundscape — is it too uniform?"""
import numpy as np
import onnxruntime as ort
import librosa
import soundfile as sf
from pathlib import Path

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


sess = ort.InferenceSession("kaggle_model/audiomae_emb_v160.onnx",
                            providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

# Sample 5 soundscapes
files = sorted(Path("data/train_soundscapes").glob("*.ogg"))[:5]
all_within = []
all_across = []
embeddings = []

for fp in files:
    wav, sr = sf.read(str(fp))
    if sr != 32000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=32000)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    target = 60 * 32000
    if len(wav) >= target:
        wav = wav[:target]
    else:
        wav = np.pad(wav, (0, target - len(wav)))
    chunks = wav.reshape(12, 5*32000).astype(np.float32)
    mel = audiomae_mel(chunks)
    emb = sess.run(None, {inp: mel})[0]  # (12, 768) L2-normed
    embeddings.append(emb)
    sim = emb @ emb.T  # (12, 12)
    # Off-diagonal within-soundscape similarity
    mask = ~np.eye(12, dtype=bool)
    within = sim[mask]
    print(f"{fp.name[:40]} within-soundscape cos: min={within.min():.3f} mean={within.mean():.3f} max={within.max():.3f}")
    all_within.extend(within.tolist())

# Across-soundscape similarity
for i in range(len(embeddings)):
    for j in range(i+1, len(embeddings)):
        sim_ij = embeddings[i] @ embeddings[j].T  # (12, 12)
        all_across.extend(sim_ij.flatten().tolist())

print(f"\nWithin-soundscape (n={len(all_within)}): mean={np.mean(all_within):.3f} std={np.std(all_within):.3f}")
print(f"Across-soundscape (n={len(all_across)}): mean={np.mean(all_across):.3f} std={np.std(all_across):.3f}")
print(f"Within - Across: {np.mean(all_within) - np.mean(all_across):+.3f}")
