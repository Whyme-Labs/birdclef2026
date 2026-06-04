"""Phase 1 (V246 prereq): Extract 5-fold public SED soft labels on all focal training audio.

Optimized for speed:
  - Parallelized via Modal .map() across N_SHARDS containers
  - Each shard processes ~ceil(N_files/N_SHARDS) files
  - librosa mel exactly matches public SED training (fmin=20, fmax=16000, power-to-db, z-score)
  - Batched ONNX inference (32 chunks per call)

Output volume (birdclef-sed-softlabels):
  - softlabels_clip_5fold.npy     (N_files, 234)  averaged 5-fold sigmoid clip probs
  - softlabels_clip_per_fold.npy  (N_files, 5, 234)
  - file_index.csv                ordered filenames
  - file_chunks_softlabels.npy    (N_chunks_total, 234) chunk-level
  - file_chunk_offsets.npy        (N_files+1,) cumulative offsets
"""
import modal

app = modal.App("birdclef-sed-soft-v2")
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

N_SHARDS = 4  # reduced from 8 — Modal preempted 8 concurrent A100s earlier; 4 should fit capacity


@app.function(
    image=image,
    gpu="A100",
    volumes={"/audio": audio_vol, "/sed": sed_vol, "/out": out_vol},
    timeout=2 * 3600,
    cpu=4,
)
def extract_shard(shard_id: int, n_shards: int = N_SHARDS):
    import os, time, numpy as np, pandas as pd
    import soundfile as sf
    import librosa
    import onnxruntime as ort
    from tqdm import tqdm

    SR = 32000
    WS = SR * 5
    N_MELS = 256
    N_FFT = 2048
    HOP = 512
    NC = 234
    FMIN = 20
    FMAX = 16000

    DATA = "/audio"
    SED_DIR = "/sed"
    OUT = "/out"

    df_full = pd.read_csv(f"{DATA}/train.csv")
    df_full["filename"] = df_full["filename"].astype(str)

    # Shard by index
    n_total = len(df_full)
    shard_size = (n_total + n_shards - 1) // n_shards
    start = shard_id * shard_size
    end = min((shard_id + 1) * shard_size, n_total)
    df = df_full.iloc[start:end].reset_index(drop=True)
    n_files = len(df)
    print(f"[shard {shard_id}/{n_shards}] processing files {start}..{end} (n={n_files})")

    # Load 5-fold ONNX sessions
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
        providers = sess.get_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(f"[shard {shard_id}] CUDA not available! providers={providers}. Aborting to avoid CPU fallback.")
    print(f"[shard {shard_id}] loaded 5 SED folds (CUDA confirmed)")
    input_name = sess_list[0].get_inputs()[0].name
    clip_out_name = None
    for o in sess_list[0].get_outputs():
        if "clip" in o.name.lower():
            clip_out_name = o.name
            break
    if clip_out_name is None:
        clip_out_name = sess_list[0].get_outputs()[0].name
    print(f"[shard {shard_id}] input={input_name}, clip_output={clip_out_name}")

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
        return s.astype(np.float32)  # (256, 313)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    clip_per_fold = np.zeros((n_files, 5, NC), dtype=np.float32)
    chunk_records = []  # list of (n_chunks, 234) per file averaged
    file_chunk_counts = np.zeros(n_files, dtype=np.int32)
    skipped_count = 0

    t0 = time.time()
    for fi, row in enumerate(df.itertuples(index=False)):
        fn = row.filename
        path = f"{DATA}/train_audio/{fn}"
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception as e:
            skipped_count += 1
            chunk_records.append(np.zeros((1, NC), dtype=np.float32))
            file_chunk_counts[fi] = 1
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)

        n = len(wav)
        if n < WS:
            wav = np.tile(wav, (WS // n) + 1)[:WS]
            n = WS

        n_chunks = min(n // WS, 24)  # cap at 2 min worth
        if n_chunks < 1:
            n_chunks = 1
        chunks = np.stack([wav[i * WS:(i + 1) * WS] for i in range(n_chunks)])

        # Compute mels
        mel_batch = np.stack([make_mel(c) for c in chunks])[:, None]  # (C, 1, 256, 313)

        # Predict with each fold (batched)
        fold_chunk_probs = np.zeros((n_chunks, 5, NC), dtype=np.float32)
        for f, sess in enumerate(sess_list):
            logits = sess.run([clip_out_name], {input_name: mel_batch})[0]  # (C, 234)
            fold_chunk_probs[:, f, :] = sigmoid(logits)

        # File-level: max over chunks (matches public kernel)
        clip_per_fold[fi] = fold_chunk_probs.max(axis=0)  # (5, 234)
        # Chunk-level: mean over folds
        chunk_records.append(fold_chunk_probs.mean(axis=1))  # (n_chunks, 234)
        file_chunk_counts[fi] = n_chunks

        if (fi + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (n_files - (fi + 1)) / rate / 60
            print(f"[shard {shard_id}] {fi+1}/{n_files} rate={rate:.1f}f/s eta={eta:.1f}min", flush=True)

        # Checkpoint every 1000 files to allow resume on preemption
        if (fi + 1) % 1000 == 0:
            try:
                out_dir = f"{OUT}/shard_{shard_id:02d}_checkpoint"
                os.makedirs(out_dir, exist_ok=True)
                np.save(f"{out_dir}/clip_per_fold_partial.npy", clip_per_fold[:fi+1])
                np.save(f"{out_dir}/chunks_partial.npy", np.concatenate(chunk_records, axis=0) if chunk_records else np.zeros((0, NC), dtype=np.float32))
                np.save(f"{out_dir}/counts_partial.npy", file_chunk_counts[:fi+1])
                with open(f"{out_dir}/progress.txt", "w") as f:
                    f.write(str(fi + 1))
                out_vol.commit()
                print(f"[shard {shard_id}] checkpoint at {fi+1}", flush=True)
            except Exception as e:
                print(f"[shard {shard_id}] checkpoint failed: {e}", flush=True)

    all_chunks = np.concatenate(chunk_records, axis=0) if chunk_records else np.zeros((0, NC), dtype=np.float32)
    file_offsets = np.zeros(n_files + 1, dtype=np.int64)
    file_offsets[1:] = np.cumsum(file_chunk_counts)

    out_dir = f"{OUT}/shard_{shard_id:02d}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/clip_per_fold.npy", clip_per_fold)
    np.save(f"{out_dir}/chunks.npy", all_chunks)
    np.save(f"{out_dir}/chunk_offsets.npy", file_offsets)
    np.save(f"{out_dir}/chunk_counts.npy", file_chunk_counts)
    df[["filename", "primary_label"]].to_csv(f"{out_dir}/file_index.csv", index=False)
    out_vol.commit()

    elapsed = time.time() - t0
    print(f"[shard {shard_id}] DONE n_files={n_files} n_chunks={len(all_chunks)} skipped={skipped_count} elapsed={elapsed/60:.1f}min", flush=True)
    return {"shard": shard_id, "n_files": n_files, "n_chunks": int(len(all_chunks)), "skipped": int(skipped_count), "elapsed_min": elapsed / 60}


@app.function(
    image=image,
    volumes={"/audio": audio_vol, "/out": out_vol},
    cpu=4,
    timeout=600,
)
def merge_shards(n_shards: int = N_SHARDS):
    import os, numpy as np, pandas as pd
    OUT = "/out"
    DATA = "/audio"

    df_full = pd.read_csv(f"{DATA}/train.csv")
    df_full["filename"] = df_full["filename"].astype(str)

    clip_parts = []
    chunks_parts = []
    counts_parts = []
    idx_parts = []
    for s in range(n_shards):
        d = f"{OUT}/shard_{s:02d}"
        if not os.path.exists(d):
            raise FileNotFoundError(f"missing {d}")
        clip_parts.append(np.load(f"{d}/clip_per_fold.npy"))
        chunks_parts.append(np.load(f"{d}/chunks.npy"))
        counts_parts.append(np.load(f"{d}/chunk_counts.npy"))
        idx_parts.append(pd.read_csv(f"{d}/file_index.csv"))

    clip_per_fold = np.concatenate(clip_parts, axis=0)
    all_chunks = np.concatenate(chunks_parts, axis=0)
    counts = np.concatenate(counts_parts, axis=0)
    df_idx = pd.concat(idx_parts, ignore_index=True)
    offsets = np.zeros(len(df_idx) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    clip_mean = clip_per_fold.mean(axis=1)
    np.save(f"{OUT}/softlabels_clip_5fold.npy", clip_mean)
    np.save(f"{OUT}/softlabels_clip_per_fold.npy", clip_per_fold)
    np.save(f"{OUT}/file_chunks_softlabels.npy", all_chunks)
    np.save(f"{OUT}/file_chunk_offsets.npy", offsets)
    np.save(f"{OUT}/chunk_counts.npy", counts)
    df_idx.to_csv(f"{OUT}/file_index.csv", index=False)

    print(f"[merge] clip_mean: {clip_mean.shape} mean_prob={clip_mean.mean():.4f}")
    print(f"[merge] chunks: {all_chunks.shape}")
    print(f"[merge] files: {len(df_idx)}")
    out_vol.commit()
    return {"n_files": len(df_idx), "n_chunks": int(len(all_chunks))}


@app.local_entrypoint()
def main(action: str = "all"):
    if action in ("extract", "all"):
        # Launch N_SHARDS in parallel
        results = list(extract_shard.starmap([(s, N_SHARDS) for s in range(N_SHARDS)]))
        print("=== Shard results ===")
        for r in results:
            print(r)
    if action in ("merge", "all"):
        m = merge_shards.remote(N_SHARDS)
        print(f"=== Merge result ===\n{m}")
