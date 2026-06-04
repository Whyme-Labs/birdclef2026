"""Phase 1b: Extract 5-fold public SED + (later) ProtoSSM pseudo-labels on UNLABELED train_soundscapes.

Per 2025 2nd place: this is the key domain-shift adaptation mechanism.
Pseudo selection logic (from paper):
  - Max-prob > 0.5: keep chunk
  - Probabilities < 0.1: zero out (suppress noise)
  - Saved as soft target vectors per chunk

Output volume (birdclef-sed-softlabels):
  - soundscape_pseudo_chunks.npy    (N_chunks, 234) post-selection soft probs
  - soundscape_pseudo_filenames.csv (N_chunks_total,) filename per chunk row
  - soundscape_chunk_offsets.npy    cumulative offsets per file

Will be merged with focal soft labels in Phase 2 student training.
"""
import modal

app = modal.App("birdclef-soundscape-pseudo")
audio_vol = modal.Volume.from_name("birdclef-audio-data", create_if_missing=False)
sed_vol = modal.Volume.from_name("birdclef-public-sed", create_if_missing=False)
out_vol = modal.Volume.from_name("birdclef-sed-softlabels", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "onnxruntime-gpu==1.20.0",
        "numpy",
        "pandas",
        "soundfile",
        "librosa==0.10.2",
        "tqdm",
    )
    .env({"LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"})
)

N_SHARDS = 4  # 10k files; 4 shards x 2.5k each


@app.function(
    image=image, gpu="A100",
    volumes={"/audio": audio_vol, "/sed": sed_vol, "/out": out_vol},
    timeout=2 * 3600, cpu=4,
)
def extract_shard(shard_id: int, n_shards: int = N_SHARDS):
    import os, glob, time, numpy as np, pandas as pd
    import soundfile as sf
    import librosa
    import onnxruntime as ort

    SR = 32000
    WS = SR * 5
    N_MELS = 256
    N_FFT = 2048
    HOP = 512
    NC = 234
    FMIN = 20
    FMAX = 16000
    PSEUDO_KEEP_THRESH = 0.5
    PSEUDO_ZERO_THRESH = 0.1

    DATA = "/audio"
    SED_DIR = "/sed"
    OUT = "/out"

    soundscape_files = sorted(glob.glob(f"{DATA}/train_soundscapes/*.ogg"))
    n_total = len(soundscape_files)
    shard_size = (n_total + n_shards - 1) // n_shards
    start = shard_id * shard_size
    end = min((shard_id + 1) * shard_size, n_total)
    files = soundscape_files[start:end]
    print(f"[shard {shard_id}/{n_shards}] processing soundscape files {start}..{end} (n={len(files)})")

    # Load 5-fold SED
    sess_list = []
    for f in range(5):
        path = f"{SED_DIR}/sed_fold{f}.onnx"
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, sess_options=so,
                                     providers=[("CUDAExecutionProvider", {"device_id": 0}),
                                                "CPUExecutionProvider"])
        sess_list.append(sess)
        if "CUDAExecutionProvider" not in sess.get_providers():
            raise RuntimeError(f"CUDA not loaded on shard {shard_id}")
    print(f"[shard {shard_id}] loaded 5 SED folds (CUDA confirmed)")
    input_name = sess_list[0].get_inputs()[0].name
    clip_out = "clip_logits"

    def make_mel(wav_chunk):
        s = librosa.feature.melspectrogram(
            y=wav_chunk, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=80)
        s = (s - s.mean()) / (s.std() + 1e-6)
        if s.shape[-1] < 313:
            s = np.pad(s, ((0, 0), (0, 313 - s.shape[-1])))
        else:
            s = s[..., :313]
        return s.astype(np.float32)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    chunk_records = []
    fname_records = []
    chunk_idx_records = []
    file_counts = []
    kept_chunks = 0

    t0 = time.time()
    for fi, path in enumerate(files):
        fn = os.path.basename(path)
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            file_counts.append(0)
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)

        n = len(wav)
        if n < WS:
            wav = np.tile(wav, (WS // n) + 1)[:WS]
            n = WS

        # Soundscapes are 60s → 12 chunks
        n_chunks = min(n // WS, 12)
        chunks = np.stack([wav[i * WS:(i + 1) * WS] for i in range(n_chunks)])
        mel_batch = np.stack([make_mel(c) for c in chunks])[:, None]  # (C, 1, 256, 313)

        fold_probs = np.zeros((n_chunks, 5, NC), dtype=np.float32)
        for f, sess in enumerate(sess_list):
            logits = sess.run([clip_out], {input_name: mel_batch})[0]
            fold_probs[:, f, :] = sigmoid(logits)
        chunk_mean = fold_probs.mean(axis=1)  # (n_chunks, 234)

        # Apply 2025-paper selection
        retained_ci = []
        retained_probs = []
        for ci in range(n_chunks):
            row = chunk_mean[ci]
            if row.max() < PSEUDO_KEEP_THRESH:
                continue  # drop chunk - no confident class
            zeroed = row.copy()
            zeroed[zeroed < PSEUDO_ZERO_THRESH] = 0.0
            retained_ci.append(ci)
            retained_probs.append(zeroed)
            kept_chunks += 1

        file_counts.append(len(retained_ci))
        for ci, p in zip(retained_ci, retained_probs):
            chunk_records.append(p)
            fname_records.append(fn)
            chunk_idx_records.append(ci)

        if (fi + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (len(files) - (fi + 1)) / rate / 60
            print(f"[shard {shard_id}] {fi+1}/{len(files)} kept={kept_chunks} rate={rate:.1f}f/s eta={eta:.1f}min", flush=True)

    pseudo_arr = np.stack(chunk_records, axis=0).astype(np.float32) if chunk_records else np.zeros((0, NC), dtype=np.float32)
    df_out = pd.DataFrame({"filename": fname_records, "chunk_idx": chunk_idx_records})

    out_dir = f"{OUT}/soundscape_shard_{shard_id:02d}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/pseudo_chunks.npy", pseudo_arr)
    df_out.to_csv(f"{out_dir}/pseudo_meta.csv", index=False)
    out_vol.commit()

    elapsed = time.time() - t0
    print(f"[shard {shard_id}] DONE n_files={len(files)} kept_chunks={kept_chunks} elapsed={elapsed/60:.1f}min", flush=True)
    return {"shard": shard_id, "n_files": len(files), "kept_chunks": int(kept_chunks)}


@app.function(image=image, volumes={"/out": out_vol}, cpu=4, timeout=600)
def merge_shards(n_shards: int = N_SHARDS):
    import os, numpy as np, pandas as pd
    OUT = "/out"

    chunk_parts = []
    meta_parts = []
    for s in range(n_shards):
        d = f"{OUT}/soundscape_shard_{s:02d}"
        if not os.path.exists(d):
            raise FileNotFoundError(d)
        chunk_parts.append(np.load(f"{d}/pseudo_chunks.npy"))
        meta_parts.append(pd.read_csv(f"{d}/pseudo_meta.csv"))

    all_chunks = np.concatenate(chunk_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True)

    np.save(f"{OUT}/soundscape_pseudo_chunks.npy", all_chunks)
    meta.to_csv(f"{OUT}/soundscape_pseudo_meta.csv", index=False)
    out_vol.commit()

    print(f"[merge] {all_chunks.shape[0]} pseudo chunks across {meta['filename'].nunique()} files")
    print(f"[merge] mean prob per chunk: {all_chunks.mean():.4f}; max prob mean: {all_chunks.max(axis=1).mean():.4f}")
    return {"n_chunks": int(len(all_chunks)), "n_files": int(meta["filename"].nunique())}


@app.local_entrypoint()
def main(action: str = "all"):
    if action in ("extract", "all"):
        results = list(extract_shard.starmap([(s, N_SHARDS) for s in range(N_SHARDS)]))
        print("=== Shard results ===")
        for r in results:
            print(r)
    if action in ("merge", "all"):
        m = merge_shards.remote(N_SHARDS)
        print(f"=== Merge result ===\n{m}")
