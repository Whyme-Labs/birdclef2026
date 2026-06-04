"""V194 iter 2 prep: generate V194_iter1 pseudo-labels on the same training pool
that V137 pseudo-labeled. Output format matches pseudo_labels_v137/raw_predictions.csv
so iter 2 training can blend the two pseudo sources.

Multi-iter NS recipe step 2:
  iter 1 student (V194_iter1) → pseudo predictions on training audio
  → combined with V137 pseudo as new soft targets for iter 2 student.
"""
import argparse, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from torchaudio.transforms import MelSpectrogram

from src.config import Config
from src.models_v2 import SEDModelV2


SR = 32000
CHUNK_SEC = 5
N_MELS = 224
FMIN = 0.0
FMAX = 16000.0
N_FFT = 2048
HOP_LENGTH = 512


def make_mel(audio_chunk_32k: np.ndarray, mel_transform, db_to_amp=False):
    """Convert 5s audio at 32kHz → log-mel matching V194 training preprocessing.
    Returns (3, n_mels, T) — 3-channel via repeat to match input_chans=3.
    """
    wav = torch.from_numpy(audio_chunk_32k).float()
    spec = mel_transform(wav)  # (n_mels, T)
    log_mel = torch.log(spec.clamp(min=1e-9))
    # Normalize per-spec
    mn, mx = log_mel.min(), log_mel.max()
    if mx > mn:
        log_mel = (log_mel - mn) / (mx - mn + 1e-9)
    # Stack to 3 channels (R G B input expectation)
    out = log_mel.unsqueeze(0).repeat(3, 1, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v194_ns_iter1/best_fold0.pt")
    ap.add_argument("--source_csv", default="pseudo_labels_v137/raw_predictions.csv",
                    help="Defines which (file, end_time) chunks to predict on.")
    ap.add_argument("--out_dir", default="pseudo_labels_v194_iter1")
    ap.add_argument("--audio_root", default="data/train_audio")
    ap.add_argument("--xc_audio_root", default="data_external/xc_pantanal_audio")
    ap.add_argument("--soundscape_root", default="data/train_soundscapes")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    src = pd.read_csv(args.source_csv)
    print(f"Source pseudo: {len(src)} rows")
    label_cols = [c for c in src.columns if c not in ("file", "end_time")]
    print(f"Classes: {len(label_cols)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "raw_predictions.csv"

    # Load V194 iter 1
    print(f"Loading {args.checkpoint} on {args.device}")
    config = Config(
        backbone="tf_efficientnetv2_s.in21k",
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    model = SEDModelV2(backbone=config.backbone, num_classes=len(label_cols))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(args.device)

    mel_transform = MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        f_min=FMIN, f_max=FMAX, n_mels=N_MELS,
        power=1.0, center=True,
    )

    # Resolve audio paths — check 3 roots
    roots = [Path(args.audio_root), Path(args.xc_audio_root), Path(args.soundscape_root)]

    def find_audio(file_rel):
        # Try direct join first
        for r in roots:
            p = r / file_rel
            if p.exists():
                return p
            # Try as basename only
            p = r / Path(file_rel).name
            if p.exists():
                return p
            # Try fuzzy: any subdir
            for cand in r.rglob(Path(file_rel).name):
                return cand
        return None

    print("Phase 1: validate audio paths...")
    unique_files = src["file"].unique()
    print(f"  {len(unique_files)} unique audio files")
    found_paths = {}
    n_missing = 0
    for f in unique_files:
        p = find_audio(f)
        if p is not None:
            found_paths[f] = p
        else:
            n_missing += 1
    print(f"  found {len(found_paths)} / {len(unique_files)} ({n_missing} missing)")
    if n_missing > 0.1 * len(unique_files):
        print("  WARNING: >10% missing — check audio paths")

    src_keep = src[src["file"].isin(found_paths)].reset_index(drop=True)
    print(f"  rows to process: {len(src_keep)}")

    # Phase 2: stream batches per file
    print("\nPhase 2: extract pseudo labels...")
    t0 = time.time()
    out_rows = []
    by_file = src_keep.groupby("file", sort=False)

    audio_cache = {}
    files_done = 0
    for fn, group in by_file:
        ap_path = found_paths[fn]
        try:
            wav, sr = sf.read(str(ap_path), dtype="float32")
        except Exception as e:
            print(f"  skip {fn}: {e}")
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR)

        chunks = []
        meta_rows = []
        for _, r in group.iterrows():
            end_s = float(r["end_time"])
            start = int((end_s - CHUNK_SEC) * SR)
            end = int(end_s * SR)
            if start < 0:
                target = wav[0:max(0, end)]
                if len(target) < CHUNK_SEC * SR:
                    target = np.pad(target, (CHUNK_SEC * SR - len(target), 0))
            elif end > len(wav):
                target = wav[start:]
                if len(target) < CHUNK_SEC * SR:
                    target = np.pad(target, (0, CHUNK_SEC * SR - len(target)))
            else:
                target = wav[start:end]
            chunks.append(target.astype(np.float32))
            meta_rows.append({"file": fn, "end_time": end_s})

        # Batch through model
        for bs in range(0, len(chunks), args.batch_size):
            batch = chunks[bs:bs + args.batch_size]
            mels = torch.stack([make_mel(c, mel_transform) for c in batch]).to(args.device)
            with torch.no_grad():
                out = model(mels)
                if isinstance(out, dict):
                    out = out.get("clipwise_logit", out.get("clipwise", out.get("logit")))
                elif isinstance(out, (tuple, list)):
                    out = out[0]
                probs = torch.sigmoid(torch.clamp(out, -30, 30)).cpu().numpy()
            for k, m in enumerate(meta_rows[bs:bs + args.batch_size]):
                row = {"file": m["file"], "end_time": m["end_time"]}
                for ci, name in enumerate(label_cols):
                    row[name] = float(probs[k, ci])
                out_rows.append(row)
        files_done += 1
        if files_done % 50 == 0:
            print(f"  {files_done}/{len(by_file)} files — {time.time() - t0:.0f}s")

    print(f"\nWriting {len(out_rows)} rows to {out_csv}")
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
