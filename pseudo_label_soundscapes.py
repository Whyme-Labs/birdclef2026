"""Pseudo-label the 10,592 unlabeled train_soundscapes with the public SED 5-fold.

Output: soundscape_pseudo.npz with
  - probs    (N_windows, 234) float16  averaged 5-fold sigmoid(clip)+sigmoid(frame)
  - files    (N_windows,)     filename per window
  - win_idx  (N_windows,)     window index (0-11)

These soft labels are the soundscape-domain training target for SED v2 — the fix
for V256's focal->soundscape domain shift (V256 trained on focal, scored 0.931).
"""
import os
import glob
import time
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort

DATA = "/home/soh/birdclef-2026/data"
SED_DIR = "/home/soh/birdclef-2026/public_sed"
OUT = "/home/soh/birdclef-2026/soundscape_pseudo.npz"

SR = 32000
WS = SR * 5
N_MELS, N_FFT, HOP, FMIN, FMAX, N_TIME = 256, 2048, 512, 20, 16000, 313
N_WIN = 12
BATCH = 64


def make_mel(wav):
    s = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    s = librosa.power_to_db(s, top_db=80)
    s = (s - s.mean()) / (s.std() + 1e-6)
    if s.shape[-1] < N_TIME:
        s = np.pad(s, ((0, 0), (0, N_TIME - s.shape[-1])))
    else:
        s = s[:, :N_TIME]
    return s.astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def main():
    sessions = []
    for f in range(5):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(f"{SED_DIR}/sed_fold{f}.onnx", sess_options=so,
                                  providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        sessions.append(s)
    print(f"SED 5-fold loaded, provider: {sessions[0].get_providers()[0]}", flush=True)
    inp = sessions[0].get_inputs()[0].name

    files = sorted(glob.glob(f"{DATA}/train_soundscapes/*.ogg"))
    print(f"{len(files)} soundscape files -> {len(files)*N_WIN} windows", flush=True)

    all_probs, all_files, all_win = [], [], []
    mel_buf, meta_buf = [], []

    def flush():
        if not mel_buf:
            return
        x = np.stack(mel_buf)[:, None].astype(np.float32)  # (B,1,256,313)
        psum = np.zeros((len(mel_buf), 234), dtype=np.float32)
        for s in sessions:
            outs = s.run(None, {inp: x})
            clip = outs[0]
            frame_max = outs[1].max(axis=1) if len(outs) > 1 and outs[1].ndim == 3 else clip
            psum += 0.5 * sigmoid(clip) + 0.5 * sigmoid(frame_max)
        psum /= len(sessions)
        for k, (fn, wi) in enumerate(meta_buf):
            all_probs.append(psum[k].astype(np.float16))
            all_files.append(fn)
            all_win.append(wi)
        mel_buf.clear()
        meta_buf.clear()

    t0 = time.time()
    for fi, path in enumerate(files):
        fn = os.path.basename(path)
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"  read err {fn}: {e}", flush=True)
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        need = N_WIN * WS
        if len(wav) < need:
            wav = np.pad(wav, (0, need - len(wav)))
        for wi in range(N_WIN):
            chunk = wav[wi * WS:(wi + 1) * WS]
            mel_buf.append(make_mel(chunk))
            meta_buf.append((fn, wi))
            if len(mel_buf) >= BATCH:
                flush()
        if (fi + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {fi+1}/{len(files)} files  t={el:.0f}s  eta={el/(fi+1)*(len(files)-fi-1):.0f}s", flush=True)
    flush()

    probs = np.stack(all_probs)
    np.savez_compressed(OUT, probs=probs,
                        files=np.array(all_files), win_idx=np.array(all_win, dtype=np.int16))
    print(f"DONE: {probs.shape} pseudo-labels saved to {OUT}", flush=True)
    print(f"  mean prob {probs.astype(np.float32).mean():.4f}  "
          f"max-per-window mean {probs.astype(np.float32).max(1).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
