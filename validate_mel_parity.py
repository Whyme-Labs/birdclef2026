"""Compare numpy/librosa mel preprocessing vs HuggingFace ClapProcessor mel.

If they don't match exactly, the V174 Kaggle inference will produce wrong embeddings
and probes will fail.
"""
import numpy as np
import librosa
import soundfile as sf
import torch
from transformers import ClapProcessor, ClapModel
from pathlib import Path

# Load real soundscape audio
audio_files = list(Path("data/train_soundscapes").glob("*.ogg"))[:3]
print(f"Testing on {len(audio_files)} files")

# HF preprocessing (ground truth)
processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")
SR = 48000
NFFT = 1024
HOP = 480
NMELS = 64
FMIN = 50.0
FMAX = 14000.0
MAX_SAMPLES = 480_000
NB_FRAMES = 1001

# My numpy implementation
_window = np.hanning(NFFT).astype(np.float32)
_mel_filters = librosa.filters.mel(
    sr=SR, n_fft=NFFT, n_mels=NMELS,
    fmin=FMIN, fmax=FMAX, htk=False, norm="slaney"
).astype(np.float32)

print(f"My mel filters shape: {_mel_filters.shape}")

# HF mel filters
hf_filters = processor.feature_extractor.mel_filters_slaney
print(f"HF mel_filters_slaney shape: {hf_filters.shape}")
# HF filters are (n_freq, n_mels) = (513, 64). Mine are (n_mels, n_freq) = (64, 513). Need to transpose.
print(f"HF filter dtype: {hf_filters.dtype}")
print(f"My filter dtype: {_mel_filters.dtype}")

# Compare filter banks
_my_T = _mel_filters.T  # (513, 64) to match HF orientation
print(f"\nFilter bank L1 diff: {np.abs(_my_T - hf_filters).sum():.4f}")
print(f"Filter bank max diff: {np.abs(_my_T - hf_filters).max():.4f}")

def my_mel(wav):
    """Numpy/librosa mel preprocessing."""
    if wav.shape[0] < MAX_SAMPLES:
        n_repeat = MAX_SAMPLES // wav.shape[0]
        wav_padded = np.tile(wav, n_repeat)
        if wav_padded.shape[0] < MAX_SAMPLES:
            wav_padded = np.pad(wav_padded, (0, MAX_SAMPLES - wav_padded.shape[0]))
    else:
        wav_padded = wav[:MAX_SAMPLES]
    stft = librosa.stft(wav_padded.astype(np.float32),
                        n_fft=NFFT, hop_length=HOP,
                        win_length=NFFT, window="hann",
                        center=True, pad_mode="reflect")
    power = np.abs(stft) ** 2
    mel = _mel_filters @ power  # (n_mels, T)
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
    log_mel = log_mel.T[:NB_FRAMES, :]
    if log_mel.shape[0] < NB_FRAMES:
        log_mel = np.pad(log_mel, ((0, NB_FRAMES - log_mel.shape[0]), (0, 0)))
    return log_mel

# Test on real audio
for fp in audio_files[:3]:
    wav, sr = sf.read(str(fp))
    if sr != SR:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    # Take first 5 seconds
    chunk = wav[:5*SR].astype(np.float32)

    # HF
    hf_inputs = processor(audios=[chunk], sampling_rate=SR, return_tensors="pt")
    hf_mel = hf_inputs["input_features"][0, 0].numpy()  # (1001, 64)
    print(f"\n{fp.name}: HF mel shape={hf_mel.shape}, range=[{hf_mel.min():.2f}, {hf_mel.max():.2f}]")

    # Mine
    mine = my_mel(chunk)
    print(f"  My mel shape={mine.shape}, range=[{mine.min():.2f}, {mine.max():.2f}]")
    diff = np.abs(hf_mel - mine)
    print(f"  Diff: max={diff.max():.4f}, mean={diff.mean():.4f}, median={np.median(diff):.4f}")
    rel = np.abs(hf_mel - mine).max() / (np.abs(hf_mel).max() + 1e-9)
    print(f"  Relative max diff: {rel:.4f}")

    # Test if BioLingual output is similar with both mels
    from export_biolingual import BioLingualAudioEncoder
    # Avoid loading BioLingual model multiple times — load once
    if 'enc' not in dir():
        m = ClapModel.from_pretrained("davidrrobinson/BioLingual")
        m.eval()
        enc = BioLingualAudioEncoder(m).eval()

    with torch.no_grad():
        # HF features
        hf_feats = enc(hf_inputs["input_features"], hf_inputs.get("is_longer", torch.zeros(1, 1, dtype=torch.bool))).numpy()
        # My features
        my_feat_t = torch.from_numpy(mine[None, None, :, :].astype(np.float32))
        my_feats = enc(my_feat_t, torch.zeros(1, 1, dtype=torch.bool)).numpy()
    cos = (hf_feats[0] @ my_feats[0]) / (np.linalg.norm(hf_feats[0]) * np.linalg.norm(my_feats[0]))
    print(f"  Embedding cosine similarity (HF vs mine): {cos:.4f}")
