"""V211 Whisper-FT pseudo-label generation on V137's training pool.

Output format matches pseudo_labels_v137/raw_predictions.csv so V212 distillation
training can blend Whisper-FT pseudo with V137 pseudo (same as V194 iter 2 used
V137-pseudo + V194_iter1-pseudo blend).

Run after finetune_whisper.py completes.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import soundfile as sf
import librosa

from transformers import WhisperModel, WhisperFeatureExtractor


WHISPER_MODEL = "openai/whisper-small"
WHISPER_SR = 16000
SR_NATIVE = 32000
CHUNK_SEC = 5
N_CLASSES = 234


class WhisperFTClassifier(nn.Module):
    def __init__(self, model_id=WHISPER_MODEL, n_classes=N_CLASSES, dropout=0.2):
        super().__init__()
        full_model = WhisperModel.from_pretrained(model_id)
        self.encoder = full_model.encoder
        d = self.encoder.config.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, n_classes),
        )

    def forward(self, mel):
        enc = self.encoder.base_model.model if hasattr(self.encoder, "base_model") else self.encoder
        out = enc(input_features=mel)
        return self.head(out.last_hidden_state.mean(dim=1))


def load_finetuned(checkpoint_path: str, lora_r=16, lora_alpha=32):
    from peft import LoraConfig, get_peft_model, TaskType
    model = WhisperFTClassifier()
    cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.0, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model.encoder = get_peft_model(model.encoder, cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"    missing[:3]: {missing[:3]}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v211_whisper_small_lora/epoch_5.pt")
    ap.add_argument("--source_csv", default="pseudo_labels_v137/raw_predictions.csv")
    ap.add_argument("--out_dir", default="pseudo_labels_whisper_ft")
    ap.add_argument("--audio_root", default="data/train_audio")
    ap.add_argument("--xc_audio_root", default="data_external/xc_pantanal_audio")
    ap.add_argument("--soundscape_root", default="data/train_soundscapes")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint {args.checkpoint} not found")
        print("Wait for finetune_whisper.py to complete first.")
        sys.exit(1)

    src = pd.read_csv(args.source_csv)
    print(f"Source: {len(src)} rows")
    label_cols = [c for c in src.columns if c not in ("file", "end_time")]
    assert len(label_cols) == N_CLASSES, f"Expected {N_CLASSES} label cols, got {len(label_cols)}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "raw_predictions.csv"

    print(f"Loading Whisper-FT from {args.checkpoint} on {args.device}...")
    t0 = time.time()
    model = load_finetuned(args.checkpoint)
    model.eval().to(args.device)
    print(f"  loaded in {time.time() - t0:.1f}s")

    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)

    roots = [Path(args.audio_root), Path(args.xc_audio_root), Path(args.soundscape_root)]

    def find_audio(file_rel):
        for r in roots:
            p = r / file_rel
            if p.exists():
                return p
            p = r / Path(file_rel).name
            if p.exists():
                return p
            for cand in r.rglob(Path(file_rel).name):
                return cand
        return None

    print("\nValidate audio paths...")
    unique_files = src["file"].unique()
    found_paths = {}
    for f in unique_files:
        p = find_audio(f)
        if p is not None:
            found_paths[f] = p
    print(f"  found {len(found_paths)} / {len(unique_files)}")
    src_keep = src[src["file"].isin(found_paths)].reset_index(drop=True)
    print(f"  rows to process: {len(src_keep)}")

    print("\nExtracting Whisper-FT pseudo predictions...")
    t0 = time.time()
    out_rows = []
    by_file = src_keep.groupby("file", sort=False)
    files_done = 0

    for fn, group in by_file:
        try:
            wav, sr = sf.read(str(found_paths[fn]), dtype="float32")
        except Exception as e:
            print(f"  skip {fn}: {e}")
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != WHISPER_SR:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr,
                                    target_sr=WHISPER_SR)

        chunks = []
        meta = []
        for _, r in group.iterrows():
            end_s = float(r["end_time"])
            start = int((end_s - CHUNK_SEC) * WHISPER_SR)
            end = int(end_s * WHISPER_SR)
            target_n = CHUNK_SEC * WHISPER_SR
            if start < 0:
                target = wav[0:max(0, end)]
                if len(target) < target_n:
                    target = np.pad(target, (target_n - len(target), 0))
            elif end > len(wav):
                target = wav[start:]
                if len(target) < target_n:
                    target = np.pad(target, (0, target_n - len(target)))
            else:
                target = wav[start:end]
            chunks.append(target.astype(np.float32))
            meta.append({"file": fn, "end_time": end_s})

        for bs in range(0, len(chunks), args.batch_size):
            batch_wavs = chunks[bs:bs + args.batch_size]
            mels = []
            for w in batch_wavs:
                mel = fe(w, sampling_rate=WHISPER_SR,
                         return_tensors="pt")["input_features"][0]
                mels.append(mel)
            mel_batch = torch.stack(mels).to(args.device)
            with torch.no_grad():
                logits = model(mel_batch)
                probs = torch.sigmoid(torch.clamp(logits, -30, 30)).cpu().numpy()
            for k, m in enumerate(meta[bs:bs + args.batch_size]):
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
