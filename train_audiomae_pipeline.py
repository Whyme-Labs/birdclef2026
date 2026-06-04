"""
AudioMAE Pipeline: Domain Adaptation → Feature Extraction → Classifier

Stage 1: Self-supervised domain adaptation on unlabeled Pantanal soundscapes
          - Continue masked reconstruction on target domain
          - No labels used — no overfitting risk
          - Short training (5-10 epochs) to adapt, not overwrite AudioSet knowledge

Stage 2: Frozen feature extraction
          - Extract 768-dim embeddings from domain-adapted AudioMAE
          - One-time computation, saved to disk

Stage 3: Lightweight classifier
          - Small MLP (768 → 512 → 234) on extracted features
          - ~300K trainable params, proper ratio for 30K samples

Usage:
    python train_audiomae_pipeline.py --stage 1  # Domain adaptation
    python train_audiomae_pipeline.py --stage 2  # Feature extraction
    python train_audiomae_pipeline.py --stage 3  # Train classifier
    python train_audiomae_pipeline.py --stage all  # Run all stages
"""
import os, sys, time, math, ast, glob, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import timm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.losses import AsymmetricLoss
from src.utils import set_seed, macro_auc


# ── Mel transform matching AudioMAE expectations ────────────────────────
MEL_CONFIG = dict(sr=32000, n_fft=2048, hop_length=500, n_mels=128, fmin=20, fmax=16000)
CHUNK_SAMPLES = int(MEL_CONFIG["sr"] * 5.0)


def make_mel(waveform, mel_transform, db_transform):
    """Convert waveform to normalized mel, transposed for AudioMAE (1, time, freq)."""
    mel = db_transform(mel_transform(waveform))
    mel_min, mel_max = mel.min(), mel.max()
    mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
    mel = mel.permute(0, 2, 1)  # (1, time, freq=128)
    # Trim to multiple of 16 (patch size)
    T_dim = mel.shape[1]
    mel = mel[:, :(T_dim // 16) * 16, :]
    return mel


# ── Stage 1: Self-supervised domain adaptation ──────────────────────────
class SoundscapeMAEDataset(Dataset):
    """Random 5s chunks from unlabeled soundscapes for self-supervised MAE."""
    def __init__(self, audio_files, chunks_per_file=2):
        self.files = audio_files
        self.chunks_per_file = chunks_per_file
        self.mel_transform = T.MelSpectrogram(
            sample_rate=MEL_CONFIG["sr"], n_fft=MEL_CONFIG["n_fft"],
            hop_length=MEL_CONFIG["hop_length"], n_mels=MEL_CONFIG["n_mels"],
            f_min=MEL_CONFIG["fmin"], f_max=MEL_CONFIG["fmax"], power=2.0,
        )
        self.db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)

    def __len__(self):
        return len(self.files) * self.chunks_per_file

    def __getitem__(self, idx):
        path = self.files[idx // self.chunks_per_file]
        try:
            info = torchaudio.info(path)
            num_frames = info.num_frames
            max_start = max(0, num_frames - CHUNK_SAMPLES)
            start = random.randint(0, max_start) if max_start > 0 else 0
            waveform, sr = torchaudio.load(path, frame_offset=start, num_frames=CHUNK_SAMPLES)
            if sr != MEL_CONFIG["sr"]:
                waveform = T.Resample(sr, MEL_CONFIG["sr"])(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if waveform.shape[1] < CHUNK_SAMPLES:
                waveform = nn.functional.pad(waveform, (0, CHUNK_SAMPLES - waveform.shape[1]))
            waveform = waveform[:, :CHUNK_SAMPLES]
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak
        except Exception:
            waveform = torch.zeros(1, CHUNK_SAMPLES)
        return make_mel(waveform, self.mel_transform, self.db_transform)


class AudioMAEForMAE(nn.Module):
    """Wrap timm ViT for masked reconstruction (MAE-style)."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),  # Reconstruct patch of 16x16=256 pixels
        )

    def forward(self, x, mask_ratio=0.75):
        B = x.shape[0]
        # Get patch embeddings
        x_patches = self.backbone.patch_embed(x)
        N = x_patches.shape[1]

        # Add positional embeddings (handling dynamic size)
        if hasattr(self.backbone, 'pos_embed'):
            pos = self.backbone._pos_embed(x_patches)
            x_patches = pos

        # Random masking
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        # Keep visible patches
        x_visible = torch.gather(x_patches, dim=1,
                                  index=ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Encode visible patches
        if hasattr(self.backbone, 'cls_token'):
            cls = self.backbone.cls_token.expand(B, -1, -1)
            x_visible = torch.cat([cls, x_visible], dim=1)

        for blk in self.backbone.blocks:
            x_visible = blk(x_visible)
        x_visible = self.backbone.norm(x_visible)

        # Skip cls for reconstruction
        if hasattr(self.backbone, 'cls_token'):
            encoded = x_visible[:, 1:]
        else:
            encoded = x_visible

        # Decode visible patches
        pred = self.decoder(encoded)

        # Compute target patches
        with torch.no_grad():
            patches = x.unfold(2, 16, 16).unfold(3, 16, 16)  # (B, 1, h_patches, w_patches, 16, 16)
            patches = patches.reshape(B, -1, 16 * 16)  # (B, N, 256)
            # Normalize target
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, keepdim=True)
            patches = (patches - mean) / (var + 1e-6).sqrt()

        # Loss only on visible (we predict visible, check against kept patches)
        target_kept = torch.gather(patches, dim=1,
                                    index=ids_keep.unsqueeze(-1).expand(-1, -1, 256))
        loss = ((pred - target_kept) ** 2).mean()

        return loss


def run_stage1(args):
    """Self-supervised domain adaptation on Pantanal soundscapes."""
    print("=" * 60)
    print("Stage 1: Self-supervised domain adaptation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("data")

    # Load soundscape files
    sl_files = sorted(glob.glob(str(data_dir / "train_soundscapes" / "*.ogg")))
    print(f"Soundscape files: {len(sl_files)}")

    dataset = SoundscapeMAEDataset(sl_files, chunks_per_file=2)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)

    # Load pretrained AudioMAE backbone
    backbone = timm.create_model(
        'hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m',
        pretrained=True, num_classes=0, dynamic_img_size=True,
    )

    model = AudioMAEForMAE(backbone).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Very low LR to avoid destroying pretrained features
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    total_steps = args.adapt_epochs * len(loader)
    warmup_steps = len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        s / max(warmup_steps, 1) if s < warmup_steps else
        0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(total_steps - warmup_steps, 1))))

    out_dir = Path("checkpoints/audiomae_adapted")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.adapt_epochs):
        model.train()
        losses = []
        t0 = time.time()
        for batch_idx, mel in enumerate(loader):
            mel = mel.to(device)
            with autocast("cuda", enabled=device.type == "cuda"):
                loss = model(mel, mask_ratio=0.75)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            losses.append(loss.item())

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.adapt_epochs} | loss={np.mean(losses):.4f} | {elapsed:.0f}s")

    # Save adapted backbone
    torch.save({"backbone_state_dict": model.backbone.state_dict()},
               out_dir / "adapted_backbone.pt")
    print(f"Saved adapted backbone to {out_dir / 'adapted_backbone.pt'}")


# ── Stage 2: Feature extraction ─────────────────────────────────────────
def run_stage2(args):
    """Extract frozen features from adapted AudioMAE."""
    print("=" * 60)
    print("Stage 2: Feature extraction")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("data")
    out_dir = Path("checkpoints/audiomae_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load backbone
    backbone = timm.create_model(
        'hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m',
        pretrained=True, num_classes=0, dynamic_img_size=True,
    )
    adapted_path = Path("checkpoints/audiomae_adapted/adapted_backbone.pt")
    if adapted_path.exists():
        ckpt = torch.load(adapted_path, map_location="cpu", weights_only=False)
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        print("Loaded domain-adapted backbone")
    else:
        print("Using original AudioMAE backbone (no adaptation)")

    backbone.to(device).eval()

    mel_transform = T.MelSpectrogram(
        sample_rate=MEL_CONFIG["sr"], n_fft=MEL_CONFIG["n_fft"],
        hop_length=MEL_CONFIG["hop_length"], n_mels=MEL_CONFIG["n_mels"],
        f_min=MEL_CONFIG["fmin"], f_max=MEL_CONFIG["fmax"], power=2.0,
    )
    db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)

    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())
    label_to_idx = {l: i for i, l in enumerate(label_cols)}

    # Extract features for training recordings
    df = pd.read_csv(data_dir / "train.csv")
    df["primary_label"] = df["primary_label"].astype(str)

    features_list = []
    labels_list = []

    print(f"Extracting features for {len(df)} recordings...")
    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows()):
            path = data_dir / "train_audio" / str(row["primary_label"]) / row["filename"]
            if not path.exists():
                path = data_dir / "train_audio" / row["filename"]

            try:
                waveform, sr = torchaudio.load(path)
                if sr != MEL_CONFIG["sr"]:
                    waveform = T.Resample(sr, MEL_CONFIG["sr"])(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                # Center crop
                if waveform.shape[1] > CHUNK_SAMPLES:
                    start = (waveform.shape[1] - CHUNK_SAMPLES) // 2
                    waveform = waveform[:, start:start + CHUNK_SAMPLES]
                else:
                    waveform = nn.functional.pad(waveform, (0, CHUNK_SAMPLES - waveform.shape[1]))
                peak = waveform.abs().max()
                if peak > 0:
                    waveform = waveform / peak
            except Exception:
                waveform = torch.zeros(1, CHUNK_SAMPLES)

            mel = make_mel(waveform, mel_transform, db_transform)
            mel = mel.unsqueeze(0).to(device)  # (1, 1, T, 128)
            feat = backbone(mel).detach().cpu().numpy()  # (1, 768)
            features_list.append(feat[0])

            # Labels
            label = np.zeros(len(label_cols), dtype=np.float32)
            primary = str(row["primary_label"])
            if primary in label_to_idx:
                label[label_to_idx[primary]] = 1.0
            if "secondary_labels" in row and pd.notna(row["secondary_labels"]):
                try:
                    for sl in ast.literal_eval(str(row["secondary_labels"])):
                        sl = str(sl)
                        if sl in label_to_idx:
                            label[sl] = 1.0
                except:
                    pass
            labels_list.append(label)

            if (i + 1) % 5000 == 0:
                print(f"  [{i+1}/{len(df)}]")

    features = np.array(features_list)
    labels = np.array(labels_list)
    np.save(out_dir / "train_features.npy", features)
    np.save(out_dir / "train_labels.npy", labels)
    print(f"Saved train features: {features.shape}")

    # Extract features for soundscape windows
    sl_path = data_dir / "train_soundscapes_labels.csv"
    if sl_path.exists():
        sl_df = pd.read_csv(sl_path)
        sl_feats, sl_labels = [], []

        print(f"Extracting features for {len(sl_df)} soundscape windows...")
        for i, (_, row) in enumerate(sl_df.iterrows()):
            path = data_dir / "train_soundscapes" / row["filename"]
            start_parts = str(row["start"]).split(":")
            start_sec = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])
            start_sample = start_sec * MEL_CONFIG["sr"]

            try:
                waveform, sr = torchaudio.load(path, frame_offset=start_sample,
                                                num_frames=CHUNK_SAMPLES)
                if sr != MEL_CONFIG["sr"]:
                    waveform = T.Resample(sr, MEL_CONFIG["sr"])(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if waveform.shape[1] < CHUNK_SAMPLES:
                    waveform = nn.functional.pad(waveform, (0, CHUNK_SAMPLES - waveform.shape[1]))
                waveform = waveform[:, :CHUNK_SAMPLES]
                peak = waveform.abs().max()
                if peak > 0:
                    waveform = waveform / peak
            except Exception:
                waveform = torch.zeros(1, CHUNK_SAMPLES)

            mel = make_mel(waveform, mel_transform, db_transform)
            mel = mel.unsqueeze(0).to(device)
            feat = backbone(mel).detach().cpu().numpy()
            sl_feats.append(feat[0])

            label = np.zeros(len(label_cols), dtype=np.float32)
            for sp in str(row["primary_label"]).split(";"):
                sp = sp.strip()
                if sp in label_to_idx:
                    label[label_to_idx[sp]] = 1.0
            sl_labels.append(label)

        sl_features = np.array(sl_feats)
        sl_labels_arr = np.array(sl_labels)
        np.save(out_dir / "soundscape_features.npy", sl_features)
        np.save(out_dir / "soundscape_labels.npy", sl_labels_arr)
        print(f"Saved soundscape features: {sl_features.shape}")


# ── Stage 3: Lightweight classifier ─────────────────────────────────────
class FeatureClassifier(nn.Module):
    """Small MLP on extracted features. ~300K params."""
    def __init__(self, in_dim=768, hidden_dim=512, num_classes=234, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        logits = self.net(x)
        return {"clipwise_logit": logits, "clipwise_prob": torch.sigmoid(logits)}


class FeatureDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        label = self.labels[idx]
        if self.augment:
            # Feature-space noise augmentation
            feat = feat + torch.randn_like(feat) * 0.01
        return feat, label


def run_stage3(args):
    """Train lightweight classifier on extracted features."""
    print("=" * 60)
    print("Stage 3: Lightweight classifier training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = Path("checkpoints/audiomae_features")

    features = np.load(feat_dir / "train_features.npy")
    labels = np.load(feat_dir / "train_labels.npy")
    print(f"Train features: {features.shape}, labels: {labels.shape}")

    # Add soundscape features
    sl_feat_path = feat_dir / "soundscape_features.npy"
    if sl_feat_path.exists():
        sl_features = np.load(sl_feat_path)
        sl_labels = np.load(feat_dir / "soundscape_labels.npy")
        print(f"Soundscape features: {sl_features.shape}")
        all_features = np.concatenate([features, sl_features])
        all_labels = np.concatenate([labels, sl_labels])
    else:
        all_features, all_labels = features, labels

    # Split
    data_dir = Path("data")
    df = pd.read_csv(data_dir / "train.csv")
    df["primary_label"] = df["primary_label"].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(df, df["primary_label"]))[args.fold]

    # Soundscape indices come after train indices
    n_train = len(features)
    sl_indices = list(range(n_train, len(all_features)))

    train_indices = list(train_idx) + sl_indices
    val_indices = list(val_idx)

    train_ds = FeatureDataset(all_features[train_indices], all_labels[train_indices], augment=True)
    val_ds = FeatureDataset(all_features[val_indices], all_labels[val_indices], augment=False)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = FeatureClassifier(
        in_dim=features.shape[1], hidden_dim=512,
        num_classes=labels.shape[1], dropout=0.3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Classifier params: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    total_steps = args.clf_epochs * len(train_loader)
    warmup_steps = 2 * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        s / max(warmup_steps, 1) if s < warmup_steps else
        0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(total_steps - warmup_steps, 1))))

    loss_fn = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip_margin=0.05)

    out_dir = Path("checkpoints/audiomae_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    for epoch in range(args.clf_epochs):
        model.train()
        losses = []
        for feat, label in train_loader:
            feat, label = feat.to(device), label.to(device)
            # Mixup in feature space
            if random.random() < 0.5:
                lam = np.random.beta(0.3, 0.3)
                idx = torch.randperm(feat.size(0))
                feat = lam * feat + (1 - lam) * feat[idx]
                label = lam * label + (1 - lam) * label[idx]

            out = model(feat)
            loss = loss_fn(out["clipwise_logit"], label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            losses.append(loss.item())

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feat, label in val_loader:
                feat = feat.to(device)
                probs = model(feat)["clipwise_prob"].cpu().numpy()
                all_preds.append(probs)
                all_labels.append(label.numpy())
        val_auc = macro_auc(np.concatenate(all_labels), np.concatenate(all_preds))

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1, "val_auc": val_auc,
            }, out_dir / f"best_fold{args.fold}.pt")

        print(f"Epoch {epoch+1:02d}/{args.clf_epochs} | loss={np.mean(losses):.4f} | "
              f"val_auc={val_auc:.4f}{' *BEST*' if is_best else ''}")

    print(f"\nBest: val_auc={best_auc:.4f}")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--adapt_epochs", type=int, default=5)
    parser.add_argument("--clf_epochs", type=int, default=50)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    set_seed(42)

    if args.stage in ("1", "all"):
        run_stage1(args)
    if args.stage in ("2", "all"):
        run_stage2(args)
    if args.stage in ("3", "all"):
        run_stage3(args)


if __name__ == "__main__":
    main()
