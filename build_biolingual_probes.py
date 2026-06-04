"""V174: train per-class linear probes in BioLingual space using labeled soundscape data.

Same recipe as V162 (AudioMAE probes) but in BioLingual's 512-d audio embedding space.
BioLingual is a CLAP-style bioacoustic model trained on iNat/Macaulay/Xeno-Canto with
text supervision — different feature space than ImageNet-pretrained CNNs (Perch, EffV2S, NFNet)
and self-supervised AudioSet (AudioMAE).

Outputs:
  • kaggle_model/biolingual_probes_v174.npz — per-class linear weights and biases
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from transformers import ClapModel, ClapProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundscape_dir", default="data/train_soundscapes")
    ap.add_argument("--soundscape_csv", default="data/train_soundscapes_labels.csv")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--out", default="kaggle_model/biolingual_probes_v174.npz")
    ap.add_argument("--emb_out", default="kaggle_model/biolingual_train_emb.npz")
    ap.add_argument("--min_pos", type=int, default=3)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    tax = pd.read_csv(args.taxonomy)
    label_cols = tax["primary_label"].astype(str).tolist()
    l2i = {c: i for i, c in enumerate(label_cols)}
    print(f"{len(label_cols)} classes")

    # Build per-chunk label union (multi-label aggregation across rows in same chunk)
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
    print(f"unique chunks: {len(grouped)}")

    audio_dir = Path(args.soundscape_dir)
    existing_files = set(p.name for p in audio_dir.glob("*.ogg"))
    grouped = grouped[grouped["filename"].isin(existing_files)].reset_index(drop=True)
    print(f"chunks with audio: {len(grouped)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading BioLingual on {device}...")
    model = ClapModel.from_pretrained("davidrrobinson/BioLingual").to(device)
    processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")
    model.eval()
    SR = processor.feature_extractor.sampling_rate  # 48000

    n_chunks = len(grouped)
    X = np.zeros((n_chunks, 512), dtype=np.float32)
    Y = np.zeros((n_chunks, len(label_cols)), dtype=np.uint8)

    print(f"\nExtracting BioLingual embeddings (sr={SR}, batch={args.batch})...")
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
        if sr != SR:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        chunk_audios, slot_idx = [], []
        for _, row in group.iterrows():
            end_sec = row["end_sec"]
            start = (end_sec - 5) * SR
            end = end_sec * SR
            if start < 0:
                target = wav[0:max(0, end)]
                if len(target) < 5*SR:
                    target = np.pad(target, (5*SR - len(target), 0))
            elif end > len(wav):
                target = wav[start:]
                if len(target) < 5*SR:
                    target = np.pad(target, (0, 5*SR - len(target)))
            else:
                target = wav[start:end]
            chunk_audios.append(target.astype(np.float32))
            slot_idx.append(grouped.index.get_loc(row.name))

        # Batch through model
        for bs in range(0, len(chunk_audios), args.batch):
            batch_audio = chunk_audios[bs:bs+args.batch]
            batch_slots = slot_idx[bs:bs+args.batch]
            with torch.no_grad():
                inputs = processor(audios=batch_audio, sampling_rate=SR, return_tensors="pt").to(device)
                feats = model.get_audio_features(**inputs)
                feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()
            for k, slot in enumerate(batch_slots):
                X[slot] = feats[k]

        # Fill labels
        for _, row in group.iterrows():
            slot = grouped.index.get_loc(row.name)
            for lbl in row["primary_label"]:
                if lbl in l2i:
                    Y[slot, l2i[lbl]] = 1

        files_done += 1
        if files_done % 5 == 0:
            print(f"  {files_done}/{len(by_file)} files — {time.time()-t0:.0f}s")

    print(f"Embeddings done: {time.time()-t0:.0f}s, X={X.shape} Y={Y.shape}")
    pos_counts = Y.sum(axis=0)
    print(f"Y per-class positives: min={pos_counts.min()} median={int(np.median(pos_counts))} max={pos_counts.max()}")
    print(f"classes with ≥{args.min_pos} positives: {(pos_counts >= args.min_pos).sum()}")

    # Save embeddings (useful for tuning C/regularization without recomputing)
    Path(args.emb_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.emb_out, X=X, Y=Y, label_cols=np.array(label_cols))
    print(f"Saved embeddings: {args.emb_out}")

    # Train probes
    print("\nTraining probes...")
    weights = np.zeros((len(label_cols), 512), dtype=np.float32)
    biases = np.zeros(len(label_cols), dtype=np.float32)
    mask = np.zeros(len(label_cols), dtype=bool)

    for ci in range(len(label_cols)):
        n_pos = int(pos_counts[ci])
        if n_pos < args.min_pos:
            continue
        y = Y[:, ci].astype(int)
        try:
            clf = LogisticRegression(C=args.C, max_iter=200, solver="liblinear", class_weight="balanced")
            clf.fit(X, y)
            weights[ci] = clf.coef_[0]
            biases[ci] = clf.intercept_[0]
            mask[ci] = True
        except Exception as e:
            print(f"  skip class {label_cols[ci]} ({n_pos} pos): {e}")

    print(f"Probes trained: {mask.sum()} / {len(label_cols)}")

    np.savez_compressed(args.out,
                        weights=weights, biases=biases, mask=mask,
                        label_cols=np.array(label_cols),
                        pos_counts=pos_counts)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
