"""V182 Phase 1: Bird-MAE-Base linear probes on labeled soundscape.

Same recipe as V162 AudioMAE probes (which held 0.941 identity), but with
Bird-MAE-Base — the bird-domain MAE checkpoint that beats generic AudioMAE
by +10-16 mAP on BirdSet tasks (Rauch et al., TMLR 08/2025).

Bird-MAE input: (B, 1, 512, 128) mel from kaldi fbank, 32kHz, hop=320 (10ms),
128 mels, normalized to mean=-7.2 std=4.43.

Output: probes saved to kaggle_model/birdmae_probes_v182.npz
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import soundfile as sf
import librosa
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel


def fbank_mel(waveform, sampling_rate=32000, num_mel_bins=128, frame_shift=10,
              window_type="hanning", htk_compat=True, mean=-7.2, std=4.43,
              target_length=512):
    """Bird-MAE compatible mel via kaldi fbank.

    waveform: tensor (1, T) at sampling_rate
    Returns: (target_length, num_mel_bins) normalized
    """
    fbank = kaldi.fbank(
        waveform,
        htk_compat=htk_compat,
        sample_frequency=sampling_rate,
        use_energy=False,
        window_type=window_type,
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_shift=frame_shift,
    )
    # Pad/truncate to target_length
    n_frames = fbank.shape[0]
    if n_frames < target_length:
        pad = torch.zeros(target_length - n_frames, num_mel_bins)
        fbank = torch.cat([fbank, pad], dim=0)
    elif n_frames > target_length:
        fbank = fbank[:target_length, :]
    # Normalize
    fbank = (fbank - mean) / (std * 2.0)  # Bird-MAE feature_extractor uses mean/(2*std) — verify
    # Actually their normalization is just (x-mean)/(std), confirm by looking at the source
    return fbank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundscape_dir", default="data/train_soundscapes")
    ap.add_argument("--soundscape_csv", default="data/train_soundscapes_labels.csv")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--out", default="kaggle_model/birdmae_probes_v182.npz")
    ap.add_argument("--emb_out", default="kaggle_model/birdmae_train_emb.npz")
    ap.add_argument("--min_pos", type=int, default=3)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
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
    print(f"\nLoading Bird-MAE-Base on {device}...")
    model = AutoModel.from_pretrained("DBD-research-group/Bird-MAE-Base", trust_remote_code=True).to(device)
    model.eval()
    SR = 32000

    n_chunks = len(grouped)
    X = np.zeros((n_chunks, 768), dtype=np.float32)
    Y = np.zeros((n_chunks, len(label_cols)), dtype=np.uint8)

    print(f"\nExtracting Bird-MAE embeddings (sr={SR}, batch={args.batch})...")
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
        if wav.ndim > 1: wav = wav.mean(axis=1)

        chunks_t = []
        slot_idx = []
        for _, row in group.iterrows():
            end_sec = row["end_sec"]
            start = (end_sec - 5) * SR
            end = end_sec * SR
            if start < 0:
                target = wav[0:max(0, end)]
                if len(target) < 5*SR: target = np.pad(target, (5*SR - len(target), 0))
            elif end > len(wav):
                target = wav[start:]
                if len(target) < 5*SR: target = np.pad(target, (0, 5*SR - len(target)))
            else:
                target = wav[start:end]
            wav_t = torch.from_numpy(target.astype(np.float32)).unsqueeze(0)  # (1, T)
            mel = fbank_mel(wav_t, sampling_rate=SR)  # (512, 128)
            chunks_t.append(mel)
            slot_idx.append(grouped.index.get_loc(row.name))

        # Batch through model
        for bs in range(0, len(chunks_t), args.batch):
            batch_mels = torch.stack(chunks_t[bs:bs+args.batch]).unsqueeze(1).to(device)  # (B, 1, 512, 128)
            batch_slots = slot_idx[bs:bs+args.batch]
            with torch.no_grad():
                feats = model(batch_mels).last_hidden_state.cpu().numpy()  # (B, 768)
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
    print(f"Positives: min={pos_counts.min()} median={int(np.median(pos_counts))} max={pos_counts.max()}")
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
