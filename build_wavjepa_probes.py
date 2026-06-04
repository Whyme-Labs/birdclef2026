"""V185 part 1: WavJEPA-Nat probes for the 4-probe stack.

WavJEPA-Nat (labhamlet/wavjepa-nat-base, 200M params):
  - Audio JEPA on raw waveform, naturalistic (multichannel scenes) variant
  - 16kHz binaural (2-channel) input
  - Token-level output (B, 2ch, ~498 tokens, 768)
  - We: resample 32kHz→16kHz mono → duplicate to stereo → mean-pool tokens+channels

Adds to V183 stack as 4th feature space (raw-waveform JEPA, orthogonal to MAE/CLAP).
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
import librosa
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundscape_dir", default="data/train_soundscapes")
    ap.add_argument("--soundscape_csv", default="data/train_soundscapes_labels.csv")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--out", default="kaggle_model/wavjepa_probes_v185.npz")
    ap.add_argument("--emb_out", default="kaggle_model/wavjepa_train_emb.npz")
    ap.add_argument("--min_pos", type=int, default=3)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)  # 200M params, smaller batch
    args = ap.parse_args()

    tax = pd.read_csv(args.taxonomy)
    label_cols = tax["primary_label"].astype(str).tolist()
    l2i = {c: i for i, c in enumerate(label_cols)}
    print(f"{len(label_cols)} classes")

    sc = pd.read_csv(args.soundscape_csv)
    sc["primary_label"] = sc["primary_label"].astype(str)
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)

    def parse_lbl(x):
        if pd.isna(x) or x == "nan": return set()
        return set(t.strip() for t in str(x).split(";") if t.strip())
    def union_sets(series):
        out = set()
        for s in series: out |= s
        return out

    sc["label_set"] = sc["primary_label"].apply(parse_lbl)
    grouped = sc.groupby(["filename", "end_sec"])["label_set"].apply(union_sets).reset_index()
    grouped = grouped.rename(columns={"label_set": "primary_label"})
    audio_dir = Path(args.soundscape_dir)
    existing = set(p.name for p in audio_dir.glob("*.ogg"))
    grouped = grouped[grouped["filename"].isin(existing)].reset_index(drop=True)
    print(f"chunks: {len(grouped)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading WavJEPA-Nat on {device}...")
    model = AutoModel.from_pretrained("labhamlet/wavjepa-nat-base", trust_remote_code=True).to(device)
    model.eval()
    SR_OUT = 16000  # WavJEPA expects 16kHz
    SR_IN = 32000
    CHUNK_LEN = 5 * SR_OUT  # 80000 samples for 5s at 16kHz

    n_chunks = len(grouped)
    X = np.zeros((n_chunks, 768), dtype=np.float32)
    Y = np.zeros((n_chunks, len(label_cols)), dtype=np.uint8)

    print(f"\nExtracting WavJEPA embeddings (16kHz, batch={args.batch})...")
    t0 = time.time()
    by_file = grouped.groupby("filename")
    files_done = 0
    for fn, group in by_file:
        fp = audio_dir / fn
        try:
            wav, sr = sf.read(str(fp))
        except Exception as e:
            print(f"  skip {fn}: {e}")
            continue
        if sr != SR_IN:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR_IN)
        if wav.ndim > 1: wav = wav.mean(axis=1)

        # Resample 32k → 16k
        wav_16k = librosa.resample(wav.astype(np.float32), orig_sr=SR_IN, target_sr=SR_OUT)

        chunks_t = []
        slot_idx = []
        for _, row in group.iterrows():
            end_sec = row["end_sec"]
            start = (end_sec - 5) * SR_OUT
            end = end_sec * SR_OUT
            if start < 0:
                target = wav_16k[0:max(0, end)]
                if len(target) < CHUNK_LEN: target = np.pad(target, (CHUNK_LEN - len(target), 0))
            elif end > len(wav_16k):
                target = wav_16k[start:]
                if len(target) < CHUNK_LEN: target = np.pad(target, (0, CHUNK_LEN - len(target)))
            else:
                target = wav_16k[start:end]
            wav_t = torch.from_numpy(target.astype(np.float32))
            # Duplicate mono to stereo: (T,) → (2, T)
            wav_stereo = torch.stack([wav_t, wav_t])
            chunks_t.append(wav_stereo)
            slot_idx.append(grouped.index.get_loc(row.name))

        for bs in range(0, len(chunks_t), args.batch):
            batch_t = torch.stack(chunks_t[bs:bs+args.batch]).to(device)  # (B, 2, T)
            batch_slots = slot_idx[bs:bs+args.batch]
            with torch.no_grad():
                out = model(batch_t)
                # out is tuple, [0] is (B, 2ch, tokens, 768)
                feats = out[0]
                # Mean over channels and tokens
                feats = feats.mean(dim=(1, 2)).cpu().numpy()  # (B, 768)
            for k, slot in enumerate(batch_slots):
                X[slot] = feats[k]

        for _, row in group.iterrows():
            slot = grouped.index.get_loc(row.name)
            for lbl in row["primary_label"]:
                if lbl in l2i:
                    Y[slot, l2i[lbl]] = 1

        files_done += 1
        if files_done % 5 == 0:
            print(f"  {files_done}/{len(by_file)} — {time.time()-t0:.0f}s")

    print(f"Embeddings: {time.time()-t0:.0f}s, X={X.shape} Y={Y.shape}")
    pos_counts = Y.sum(axis=0)
    print(f"≥{args.min_pos} pos: {(pos_counts >= args.min_pos).sum()}")

    Path(args.emb_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.emb_out, X=X, Y=Y, label_cols=np.array(label_cols))

    weights = np.zeros((len(label_cols), 768), dtype=np.float32)
    biases = np.zeros(len(label_cols), dtype=np.float32)
    mask = np.zeros(len(label_cols), dtype=bool)
    print("\nTraining probes...")
    for ci in range(len(label_cols)):
        if pos_counts[ci] < args.min_pos: continue
        y = Y[:, ci].astype(int)
        try:
            clf = LogisticRegression(C=args.C, max_iter=200, solver="liblinear", class_weight="balanced")
            clf.fit(X, y)
            weights[ci] = clf.coef_[0]
            biases[ci] = clf.intercept_[0]
            mask[ci] = True
        except Exception as e:
            print(f"  skip {label_cols[ci]}: {e}")

    print(f"Probes: {mask.sum()}/{len(label_cols)}")
    np.savez_compressed(args.out, weights=weights, biases=biases, mask=mask,
                        label_cols=np.array(label_cols), pos_counts=pos_counts)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
