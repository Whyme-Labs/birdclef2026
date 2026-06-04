"""
V160: Build per-species AudioMAE embedding templates from all train_audio.

For each of the 35K train_audio recordings:
  1. Read center 5s at 32 kHz, peak-normalize.
  2. Compute log-mel (n_mels=128, n_fft=2048, hop=500, fmin=20, fmax=16000),
     transpose to (time, 128), trim time to multiple of 16.
  3. Run AudioMAE backbone (ViT-base pretrained on AudioSet 2M, our finetune)
     to get a 768-d global-pool embedding.

Then group embeddings by primary_label and save:
  • templates: (234, 768) — per-species mean embedding (zeros for zero-shot species)
  • cluster_templates: (234, K=3, 768) — k-means clusters per species when ≥K samples
  • mask: (234,) boolean — which species have any templates

Output is written to kaggle_model/audiomae_templates.npz.
"""
import argparse, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import timm
from timm.layers.pos_embed import resample_abs_pos_embed
from sklearn.cluster import KMeans


def build_backbone(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    backbone = timm.create_model(
        "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m",
        pretrained=False, num_classes=0,
        img_size=(320, 128), dynamic_img_size=False,
    )
    if "backbone.pos_embed" in sd:
        old_pe = sd["backbone.pos_embed"]
        new_pe = resample_abs_pos_embed(
            old_pe, new_size=(20, 8), old_size=(64, 8), num_prefix_tokens=1,
        )
        sd["backbone.pos_embed"] = new_pe

    backbone_state = {
        k.replace("backbone.", "", 1): v
        for k, v in sd.items() if k.startswith("backbone.")
    }
    miss, unexp = backbone.load_state_dict(backbone_state, strict=False)
    if miss or unexp:
        print(f"warn: backbone load miss={len(miss)} unexp={len(unexp)}")
    backbone.eval().to(device)
    print(f"AudioMAE backbone loaded ({sum(p.numel() for p in backbone.parameters())/1e6:.0f}M params)")
    return backbone


def make_mel_transform(device):
    return (
        torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=2048, hop_length=500, n_mels=128,
            f_min=20, f_max=16000, power=2.0, norm="slaney", mel_scale="htk",
        ).to(device),
        torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(device),
    )


def load_audio_5s(path, sr=32000):
    wav, src_sr = torchaudio.load(str(path))
    if src_sr != sr:
        wav = torchaudio.functional.resample(wav, src_sr, sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    chunk = 5 * sr
    n = wav.shape[1]
    if n >= chunk:
        # Center crop
        s = (n - chunk) // 2
        wav = wav[:, s:s + chunk]
    elif n > 0:
        # Tile-pad
        reps = chunk // n + 1
        wav = wav.repeat(1, reps)[:, :chunk]
    else:
        wav = torch.zeros(1, chunk)
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    return wav.squeeze(0)  # (chunk,)


@torch.no_grad()
def batch_to_mel(batch_wav, mel_transform, db_transform):
    # batch_wav: (B, chunk)
    mel = db_transform(mel_transform(batch_wav))  # (B, 128, T)
    # min-max per sample
    mel_flat = mel.reshape(mel.shape[0], -1)
    mn = mel_flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
    mx = mel_flat.max(dim=1, keepdim=True).values.unsqueeze(-1)
    mel = (mel - mn) / (mx - mn + 1e-7)
    # transpose (128 mels axis -> last)
    mel = mel.permute(0, 2, 1)  # (B, T, 128)
    # Trim time to multiple of 16
    T = mel.shape[1]
    target = (T // 16) * 16
    mel = mel[:, :target, :]
    return mel.unsqueeze(1)  # (B, 1, T, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/audiomae_ft_v159/best_fold0.pt")
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--audio_dir", default="data/train_audio")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--out", default="kaggle_model/audiomae_templates.npz")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--n_clusters", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = build_backbone(args.ckpt, device)
    mel_t, db_t = make_mel_transform(device)

    # Labels
    tax = pd.read_csv(args.taxonomy)
    label_cols = sorted(tax["primary_label"].astype(str).tolist())
    l2i = {c: i for i, c in enumerate(label_cols)}

    df = pd.read_csv(args.train_csv)
    df["primary_label"] = df["primary_label"].astype(str)
    print(f"train_audio rows: {len(df)}, unique species: {df['primary_label'].nunique()}")

    n_total = len(df)
    embeddings = np.zeros((n_total, 768), dtype=np.float32)
    species_idx = np.full(n_total, -1, dtype=np.int32)
    valid = np.zeros(n_total, dtype=bool)

    t0 = time.time()
    batch_paths = []
    batch_idx = []
    for i, row in df.iterrows():
        sp = row["primary_label"]
        if sp not in l2i:
            continue
        path = Path(args.audio_dir) / row["filename"]
        if not path.exists():
            continue
        species_idx[i] = l2i[sp]
        batch_paths.append((i, path))
        if len(batch_paths) >= args.batch_size or i == n_total - 1:
            wavs = []
            idxs = []
            for j, p in batch_paths:
                try:
                    w = load_audio_5s(p)
                    wavs.append(w); idxs.append(j)
                except Exception as e:
                    pass
            if wavs:
                batch_wav = torch.stack(wavs).to(device)
                with torch.no_grad():
                    mel = batch_to_mel(batch_wav, mel_t, db_t)
                    feats = backbone(mel)  # (B, 768)
                feats = feats.cpu().numpy().astype(np.float32)
                for k, j in enumerate(idxs):
                    embeddings[j] = feats[k]
                    valid[j] = True
            batch_paths = []
            if (i + 1) % 1000 < args.batch_size:
                done = valid.sum()
                rate = done / max(time.time() - t0, 1)
                eta = (n_total - i - 1) / max(rate, 0.1)
                print(f"  {done}/{n_total} valid, {rate:.1f}/s, ETA {eta/60:.1f}min", flush=True)

    print(f"Total embedded: {valid.sum()}/{n_total} in {(time.time()-t0)/60:.1f} min")

    # Per-species templates
    n_classes = len(label_cols)
    mean_templates = np.zeros((n_classes, 768), dtype=np.float32)
    cluster_templates = np.zeros((n_classes, args.n_clusters, 768), dtype=np.float32)
    mask = np.zeros(n_classes, dtype=bool)
    counts = np.zeros(n_classes, dtype=np.int32)
    for c in range(n_classes):
        sel = (species_idx == c) & valid
        if sel.sum() == 0:
            continue
        embs = embeddings[sel]
        # L2-normalize before averaging (per-sample)
        embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9)
        mean_emb = embs.mean(axis=0)
        mean_emb /= np.maximum(np.linalg.norm(mean_emb), 1e-9)
        mean_templates[c] = mean_emb
        # k-means clustering for richer templates if enough samples
        K = min(args.n_clusters, embs.shape[0])
        if K >= 2:
            km = KMeans(n_clusters=K, random_state=42, n_init=5).fit(embs)
            cents = km.cluster_centers_
            # Re-L2-normalize cluster centers
            cents = cents / np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-9)
            # Pad to n_clusters if K < n_clusters by repeating mean
            if K < args.n_clusters:
                pad = np.tile(mean_emb[None, :], (args.n_clusters - K, 1))
                cents = np.concatenate([cents, pad], axis=0)
            cluster_templates[c] = cents
        else:
            cluster_templates[c] = np.tile(mean_emb[None, :], (args.n_clusters, 1))
        mask[c] = True
        counts[c] = sel.sum()

    print(f"Species with templates: {mask.sum()}/{n_classes}")
    print(f"Counts (active species): min={counts[mask].min()} median={int(np.median(counts[mask]))} max={counts[mask].max()}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        mean_templates=mean_templates,
                        cluster_templates=cluster_templates,
                        mask=mask, counts=counts,
                        label_cols=np.array(label_cols))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
