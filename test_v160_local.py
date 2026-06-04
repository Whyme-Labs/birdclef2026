"""Local sanity check for V160 inference: AudioMAE backbone + templates → per-species scores."""
import numpy as np
import onnxruntime as ort
import librosa
import soundfile as sf
from pathlib import Path
import time


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
    print("Loading templates…")
    tmpl = np.load("kaggle_model/audiomae_templates.npz", allow_pickle=True)
    mean_t = tmpl["mean_templates"]
    cluster_t = tmpl["cluster_templates"]
    mask = tmpl["mask"]
    counts = tmpl["counts"]
    label_cols = tmpl["label_cols"]
    print(f"  mean_templates: {mean_t.shape}")
    print(f"  cluster_templates: {cluster_t.shape}")
    print(f"  mask: {mask.sum()}/{len(mask)} active")
    print(f"  counts: min={counts[mask].min()} max={counts[mask].max()}")

    # Verify L2-norms
    mean_norms = np.linalg.norm(mean_t[mask], axis=1)
    cluster_norms = np.linalg.norm(cluster_t[mask].reshape(-1, 768), axis=1)
    print(f"  mean template norms: {mean_norms.min():.3f} to {mean_norms.max():.3f}")
    print(f"  cluster template norms: {cluster_norms.min():.3f} to {cluster_norms.max():.3f}")

    print("\nLoading ONNX backbone…")
    sess = ort.InferenceSession("kaggle_model/audiomae_emb_v160.onnx",
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    print(f"  input: {inp} {sess.get_inputs()[0].shape}")

    # Pick a real train_audio file as test
    print("\nReading a sample audio file…")
    import pandas as pd
    df = pd.read_csv("data/train.csv").head(1)
    fp = Path("data/train_audio") / df.iloc[0]["filename"]
    primary = df.iloc[0]["primary_label"]
    print(f"  {fp.name}, primary={primary}")
    wav, sr = sf.read(str(fp))
    if sr != 32000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=32000)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    chunk = 5 * 32000
    if len(wav) >= chunk:
        s = (len(wav) - chunk) // 2
        wav = wav[s:s + chunk]
    else:
        reps = chunk // len(wav) + 1
        wav = np.tile(wav, reps)[:chunk]
    print(f"  audio chunk: {wav.shape}")

    print("\nComputing mel + embedding…")
    t0 = time.time()
    mel_in = audiomae_mel([wav.astype(np.float32)])
    print(f"  mel shape: {mel_in.shape}")
    emb = sess.run(None, {inp: mel_in})[0]
    print(f"  emb shape: {emb.shape}, norm: {np.linalg.norm(emb[0]):.4f}")

    print("\nComputing template similarities…")
    sim_mean = emb @ mean_t.T  # (1, 234)
    cluster_flat = cluster_t.reshape(-1, 768)
    sim_clust = (emb @ cluster_flat.T).reshape(-1, mean_t.shape[0], cluster_t.shape[1]).max(axis=2)
    score = np.maximum(sim_mean, sim_clust)
    score = np.where(mask[None, :], score, score[:, mask].mean(axis=1, keepdims=True))
    score_p = 1.0 / (1.0 + np.exp(-10.0 * score))

    # Top-5 species
    top5_idx = np.argsort(-score_p[0])[:5]
    print(f"\nTop-5 predicted species (cosine sim):")
    for i in top5_idx:
        sp = label_cols[i]
        s = score[0, i]
        sp_active = "✓" if mask[i] else "✗"
        marker = " ← TRUE" if sp == primary else ""
        print(f"  {sp} ({sp_active}): cos={s:.3f} prob={score_p[0,i]:.3f}{marker}")

    true_idx = list(label_cols).index(primary)
    true_rank = (np.argsort(-score_p[0]) == true_idx).argmax() + 1
    print(f"\nTrue species '{primary}' rank: {true_rank}/234")
    print(f"\nElapsed: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
