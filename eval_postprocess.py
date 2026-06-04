"""
Local evaluation of post-processing strategies on labeled soundscapes.
Uses the Perch models (LB862 + LB872) to generate predictions, then tests
different post-processing configurations to find optimal settings.
"""
import os, glob, time, warnings
from pathlib import Path
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio.transforms as T
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────
@dataclass
class Config:
    sr: int = 32000
    chunk_duration: float = 5.0
    n_mels: int = 224
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 0
    fmax: int = 16000
    top_db: float = 80.0
    power: float = 2.0
    norm: str = "slaney"
    mel_scale: str = "htk"
    backbone: str = "tf_efficientnet_b0.ns_jft_in1k"
    num_classes: int = 234
    in_channels: int = 3
    dropout: float = 0.1
    gem_p_init: float = 3.0

    @property
    def chunk_samples(self):
        return int(self.sr * self.chunk_duration)

cfg = Config()

# ── Model ────────────────────────────────────────────────────────────────
class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)

class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        clip_logit = (att * cls).sum(dim=-1)
        return {"clipwise_prob": torch.sigmoid(clip_logit), "clipwise_logit": clip_logit}

class SEDModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=False, in_chans=cfg.in_channels,
            features_only=False, global_pool="", num_classes=0,
        )
        self.gem_pool = GEMFreqPool(p_init=cfg.gem_p_init)
        self.head = AttentionSEDHead(self.backbone.num_features, cfg.num_classes, cfg.dropout)
    def forward(self, x):
        return self.head(self.gem_pool(self.backbone(x)))

class MelSpectrogramTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax,
            power=cfg.power, norm=cfg.norm, mel_scale=cfg.mel_scale,
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)

    @torch.no_grad()
    def forward(self, waveforms):
        mel = self.db(self.mel(waveforms.float()))
        B = mel.shape[0]
        flat = mel.reshape(B, -1)
        mel_min = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel_max = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)

# ── Load models ──────────────────────────────────────────────────────────
data_dir = Path("data")
pretrained_dir = Path("pretrained")
taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
SPECIES = sorted(taxonomy["primary_label"].astype(str).tolist())
cfg.num_classes = len(SPECIES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

models = {}
for tag, path in [("baseline", pretrained_dir / "LB862.pt"), ("finetuned", pretrained_dir / "LB872.pt")]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SEDModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    models[tag] = model
    print(f"  Loaded {path.name} ({tag})")

mel_transform = MelSpectrogramTransform(cfg).to(device).eval()

# ── Load labeled soundscapes ─────────────────────────────────────────────
sl_labels = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
sl_dir = data_dir / "train_soundscapes"

# Build ground truth: per-file, per-chunk multi-hot labels
print(f"\nLabeled soundscapes: {len(sl_labels)} windows from {sl_labels['filename'].nunique()} files")

# Group by file
file_groups = sl_labels.groupby("filename")
soundscape_files = sorted(sl_labels["filename"].unique())

# ── Run inference on all soundscape files ────────────────────────────────
CHUNK = cfg.chunk_samples

def run_ensemble(mel_batch, blend_spec):
    """Run ensemble on mel batch, return raw probabilities and logits."""
    probs = np.zeros((mel_batch.shape[0], cfg.num_classes), dtype=np.float64)
    logits = np.zeros((mel_batch.shape[0], cfg.num_classes), dtype=np.float64)
    total_w = 0.0
    for tag, model in models.items():
        w = blend_spec.get("ft" if tag == "finetuned" else "base", 0.5)
        out = model(mel_batch)
        probs += w * out["clipwise_prob"].cpu().numpy()
        logits += w * out["clipwise_logit"].cpu().numpy()
        total_w += w
    return probs / total_w, logits / total_w

print("\nRunning inference on labeled soundscapes...")
all_raw_probs = {}  # {filename: (n_chunks, 234) array of raw probs}
all_raw_logits = {}

blend_spec = {"ft": 0.8, "base": 0.2}

with torch.no_grad():
    for i, fname in enumerate(soundscape_files):
        path = sl_dir / fname
        if not path.exists():
            continue
        audio, _ = librosa.load(str(path), sr=cfg.sr, mono=True)
        audio = np.nan_to_num(audio, nan=0.0).astype(np.float32)
        n_chunks = max(1, len(audio) // CHUNK)
        audio = np.pad(audio, (0, max(0, n_chunks * CHUNK - len(audio))))[:n_chunks * CHUNK]

        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        chunks = torch.from_numpy(audio.reshape(n_chunks, CHUNK)).float().to(device)
        mel = mel_transform(chunks)
        probs, logits = run_ensemble(mel, blend_spec)
        all_raw_probs[fname] = probs
        all_raw_logits[fname] = logits

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(soundscape_files)}]")

print(f"  Done. {len(all_raw_probs)} files processed.")

# ── Build ground truth arrays ────────────────────────────────────────────
species_to_idx = {sp: i for i, sp in enumerate(SPECIES)}

gt_rows = []  # (filename, chunk_idx, multi-hot label)
pred_rows = []  # matching predictions

for fname in soundscape_files:
    if fname not in all_raw_probs:
        continue
    grp = file_groups.get_group(fname).sort_values("start")
    probs = all_raw_probs[fname]

    for _, row in grp.iterrows():
        # Parse time to chunk index
        start_sec = int(row["start"].split(":")[0]) * 3600 + int(row["start"].split(":")[1]) * 60 + int(row["start"].split(":")[2])
        chunk_idx = start_sec // 5
        if chunk_idx >= probs.shape[0]:
            continue

        # Multi-hot ground truth
        label_vec = np.zeros(cfg.num_classes, dtype=np.float32)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in species_to_idx:
                label_vec[species_to_idx[sp]] = 1.0

        gt_rows.append(label_vec)
        pred_rows.append(probs[chunk_idx])

gt_array = np.array(gt_rows)
raw_pred_array = np.array(pred_rows)
print(f"\nEvaluation set: {gt_array.shape[0]} windows, {int(gt_array.sum())} positive labels")

# ── Macro AUC computation ────────────────────────────────────────────────
def macro_auc(y_true, y_pred):
    """Competition metric: macro-averaged ROC-AUC, skip classes with no positives."""
    aucs = []
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() > 0 and y_true[:, j].sum() < len(y_true):
            try:
                aucs.append(roc_auc_score(y_true[:, j], y_pred[:, j]))
            except:
                pass
    return np.mean(aucs) if aucs else 0.0

# ── Apply post-processing variants ──────────────────────────────────────
def apply_postprocess(raw_probs_dict, gt_array, pred_indices, file_max_coef, sharpen_exp, sigma):
    """Apply post-processing and compute AUC."""
    processed = raw_pred_array.copy()

    # Need to process per-file, then extract the right chunks
    idx = 0
    for fname in soundscape_files:
        if fname not in raw_probs_dict:
            continue
        grp = file_groups.get_group(fname).sort_values("start")
        file_probs = raw_probs_dict[fname].copy()
        n_chunks = file_probs.shape[0]

        # Apply file-level post-processing
        if n_chunks > 1 and file_max_coef > 0:
            file_max = file_probs.max(axis=0, keepdims=True)
            file_probs = file_probs + file_max_coef * file_max

        if n_chunks > 1 and sharpen_exp > 1.0 and sigma > 0:
            sharpened = np.power(file_probs, sharpen_exp)
            smoothed = gaussian_filter1d(sharpened, sigma=sigma, axis=0)
            file_probs = np.power(np.maximum(smoothed, 1e-10), 1.0 / sharpen_exp)

        # Extract labeled chunk predictions
        for _, row in grp.iterrows():
            start_sec = int(row["start"].split(":")[0]) * 3600 + int(row["start"].split(":")[1]) * 60 + int(row["start"].split(":")[2])
            chunk_idx = start_sec // 5
            if chunk_idx < file_probs.shape[0]:
                processed[idx] = file_probs[chunk_idx]
                idx += 1

    return macro_auc(gt_array, processed)

# ── Baseline (no post-processing) ───────────────────────────────────────
baseline_auc = macro_auc(gt_array, raw_pred_array)
print(f"\n{'='*70}")
print(f"Baseline (no post-proc):  AUC = {baseline_auc:.6f}")
print(f"{'='*70}")

# ── Grid search post-processing params ──────────────────────────────────
print(f"\nGrid search over post-processing parameters:")
print(f"{'file_max_coef':>14} {'sharpen_exp':>12} {'sigma':>8} {'AUC':>10} {'delta':>8}")
print("-" * 60)

best_auc = 0.0
best_params = None

for fmc in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]:
    for se in [1.0, 1.2, 1.5, 2.0, 2.5]:
        for sig in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]:
            if se == 1.0 and sig > 0:
                continue  # No sharpening = no smoothing matters
            if sig == 0.0 and se > 1.0:
                continue  # No smoothing = sharpening doesn't help

            auc = apply_postprocess(all_raw_probs, gt_array, None, fmc, se, sig)
            delta = auc - baseline_auc
            marker = " *" if auc > best_auc else ""
            if delta > -0.001:  # Only show competitive configs
                print(f"{fmc:14.3f} {se:12.1f} {sig:8.2f} {auc:10.6f} {delta:+8.6f}{marker}")
            if auc > best_auc:
                best_auc = auc
                best_params = (fmc, se, sig)

print(f"\n{'='*70}")
print(f"Best params: file_max_coef={best_params[0]}, sharpen_exp={best_params[1]}, sigma={best_params[2]}")
print(f"Best AUC:    {best_auc:.6f} (delta: {best_auc - baseline_auc:+.6f})")
print(f"{'='*70}")

# ── Test blend ratios ────────────────────────────────────────────────────
print(f"\nBlend ratio sweep (with best post-processing):")
print(f"{'LB872_weight':>12} {'AUC':>10}")
print("-" * 30)

for w872 in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    w862 = 1.0 - w872
    bs = {"ft": w872, "base": w862}

    # Re-run ensemble with different blend
    temp_probs = {}
    with torch.no_grad():
        for fname in soundscape_files:
            if fname not in all_raw_logits:
                continue
            path = sl_dir / fname
            if not path.exists():
                continue
            audio, _ = librosa.load(str(path), sr=cfg.sr, mono=True)
            audio = np.nan_to_num(audio, nan=0.0).astype(np.float32)
            n_chunks = max(1, len(audio) // CHUNK)
            audio = np.pad(audio, (0, max(0, n_chunks * CHUNK - len(audio))))[:n_chunks * CHUNK]
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak
            chunks = torch.from_numpy(audio.reshape(n_chunks, CHUNK)).float().to(device)
            mel = mel_transform(chunks)
            probs, _ = run_ensemble(mel, bs)
            temp_probs[fname] = probs

    auc = apply_postprocess(temp_probs, gt_array, None, best_params[0], best_params[1], best_params[2])
    print(f"{w872:12.2f} {auc:10.6f}")

# ── Test TTA ─────────────────────────────────────────────────────────────
print(f"\nTTA evaluation (best blend + best post-processing):")
for tta_name, offsets in [("No TTA", [0]), ("TTA ±1.25s", [0, 40000, -40000]), ("TTA ±0.625s", [0, 20000, -20000])]:
    tta_probs = {}
    with torch.no_grad():
        for fname in soundscape_files:
            path = sl_dir / fname
            if not path.exists():
                continue
            audio, _ = librosa.load(str(path), sr=cfg.sr, mono=True)
            audio = np.nan_to_num(audio, nan=0.0).astype(np.float32)
            n_chunks = max(1, len(audio) // CHUNK)

            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak

            acc = np.zeros((n_chunks, cfg.num_classes), dtype=np.float64)
            count = 0
            for offset in offsets:
                if offset >= 0:
                    shifted = audio[offset:]
                else:
                    shifted = np.pad(audio, (-offset, 0))
                nc = max(1, len(shifted) // CHUNK)
                if nc != n_chunks:
                    nc = n_chunks
                shifted = np.pad(shifted, (0, max(0, nc * CHUNK - len(shifted))))[:nc * CHUNK]
                chunks = torch.from_numpy(shifted.reshape(nc, CHUNK)).float().to(device)
                mel = mel_transform(chunks)
                probs, _ = run_ensemble(mel, blend_spec)
                acc += probs
                count += 1
            tta_probs[fname] = acc / count

    auc = apply_postprocess(tta_probs, gt_array, None, best_params[0], best_params[1], best_params[2])
    print(f"  {tta_name:20s}: AUC = {auc:.6f}")

print(f"\nDone.")
