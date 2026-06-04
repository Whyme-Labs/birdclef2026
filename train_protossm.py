"""
Train ProtoSSM v5 + ResBoosting on pre-computed Perch v2 embeddings.

Training pipeline:
  1. Load pre-computed Perch v2 embeddings + soundscape labels
  2. Build file-level sequences (T=12 windows per file)
  3. Train ProtoSSM with GroupKFold (groups by filename)
  4. Generate OOF predictions from ProtoSSM
  5. Train ResidualSSM on residuals
  6. Export all trained weights for Kaggle submission

Usage:
  conda run -n birdclef python train_protossm.py
  conda run -n birdclef python train_protossm.py --d_model 320 --n_ssm_layers 4 --epochs 80
"""
import os, sys, time, json, argparse, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from collections import defaultdict

from src.protossm import ProtoSSMv5, ResidualSSM, MLPProbe


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class SoundscapeSequenceDataset(Dataset):
    """
    Each sample is one 60s soundscape file = sequence of 12 windows.
    Returns embeddings, Perch logits, labels, metadata, and label mask.
    """
    def __init__(self, file_indices, embeddings, perch_logits, labels, mask,
                 site_ids, hour_ids, augment=False, mixup_alpha=0.4, cutmix_prob=0.3):
        self.file_indices = file_indices  # list of (start_idx, end_idx) per file
        self.embeddings = embeddings      # (N_total, 1536)
        self.perch_logits = perch_logits  # (N_total, 234)
        self.labels = labels              # (N_total, 234)
        self.mask = mask                  # (N_total,) — 1 if labeled, 0 if not
        self.site_ids = site_ids          # (N_files,)
        self.hour_ids = hour_ids          # (N_files,)
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        start, end = self.file_indices[idx]
        T = end - start

        emb = self.embeddings[start:end]       # (T, 1536)
        logits = self.perch_logits[start:end]   # (T, C)
        lab = self.labels[start:end]             # (T, C)
        m = self.mask[start:end]                 # (T,)
        site = self.site_ids[idx]
        hour = self.hour_ids[idx]

        # Pad to T=12 if needed
        if T < 12:
            pad = 12 - T
            emb = np.pad(emb, ((0, pad), (0, 0)))
            logits = np.pad(logits, ((0, pad), (0, 0)))
            lab = np.pad(lab, ((0, pad), (0, 0)))
            m = np.pad(m, (0, pad))

        return {
            'embeddings': torch.from_numpy(emb).float(),
            'perch_logits': torch.from_numpy(logits).float(),
            'labels': torch.from_numpy(lab).float(),
            'mask': torch.from_numpy(m).float(),
            'site_id': torch.tensor(site, dtype=torch.long),
            'hour_id': torch.tensor(hour, dtype=torch.long),
        }


def embedding_mixup(batch, alpha=0.4):
    """MixUp in embedding space. Creates synthetic polyphonic examples."""
    if alpha <= 0:
        return batch
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5

    B = batch['embeddings'].shape[0]
    idx = torch.randperm(B)

    batch['embeddings'] = lam * batch['embeddings'] + (1 - lam) * batch['embeddings'][idx]
    batch['perch_logits'] = lam * batch['perch_logits'] + (1 - lam) * batch['perch_logits'][idx]
    # For labels: union (max) rather than interpolation, since species either present or not
    # But soft MixUp on labels also works for AUC optimization
    batch['labels'] = lam * batch['labels'] + (1 - lam) * batch['labels'][idx]
    batch['mask'] = torch.maximum(batch['mask'], batch['mask'][idx])

    return batch


# ═══════════════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

class SpeciesFocalLoss(nn.Module):
    """
    Focal loss with per-species frequency weighting.

    L = -w_c × (1-p_t)^γ × log(p_t)

    where w_c = 1/sqrt(freq_c), normalized to mean=1, capped at max_weight.
    The focal modulator (1-p_t)^γ down-weights easy examples.
    """
    def __init__(self, gamma=2.5, class_weights=None, label_smoothing=0.03):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: (B, T, C) raw logits
            targets: (B, T, C) binary labels
            mask: (B, T) — 1 for labeled windows, 0 for unlabeled
        """
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (B, T, C)

        # Focal modulator
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        # Per-species weighting
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(loss.device)

        # Mask: only compute loss on labeled windows
        if mask is not None:
            mask_expand = mask.unsqueeze(-1)  # (B, T, 1)
            loss = (loss * mask_expand).sum() / (mask_expand.sum() * loss.shape[-1] + 1e-8)
        else:
            loss = loss.mean()

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(args):
    """Load and align Perch embeddings with soundscape labels."""
    # Load embeddings
    data = np.load(args.perch_npz)
    all_emb = data['emb_full']           # (N, 1536)
    all_scores = data['scores_full_raw']  # (N, 234)
    meta = pd.read_parquet(args.perch_meta)

    # Load labels
    taxonomy = pd.read_csv('data/taxonomy.csv')
    label_cols = sorted(taxonomy['primary_label'].astype(str).tolist())
    label_to_idx = {l: i for i, l in enumerate(label_cols)}
    num_classes = len(label_cols)

    labels_df = pd.read_csv('data/train_soundscapes_labels.csv')

    # Parse labels: row_id → multi-hot vector
    def parse_start(s):
        parts = s.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    labels_df['end_sec'] = labels_df['start'].apply(lambda s: parse_start(s) + 5)
    labels_df['row_id'] = (labels_df['filename'].str.replace('.ogg', '', regex=False)
                           + '_' + labels_df['end_sec'].astype(str))

    # Build row_id → label mapping
    label_map = {}
    for _, row in labels_df.iterrows():
        rid = row['row_id']
        species = [s.strip() for s in str(row['primary_label']).split(';')]
        vec = np.zeros(num_classes, dtype=np.float32)
        for sp in species:
            if sp in label_to_idx:
                vec[label_to_idx[sp]] = 1.0
        label_map[rid] = vec

    # Build file-level sequences
    # Group by filename, each file = sequence of T windows
    site_map = {s: i for i, s in enumerate(sorted(meta['site'].unique()))}
    n_sites = len(site_map)

    files = []
    file_groups = meta.groupby('filename')
    all_labels = np.zeros((len(meta), num_classes), dtype=np.float32)
    all_mask = np.zeros(len(meta), dtype=np.float32)
    file_indices = []
    file_names = []
    file_sites = []
    file_hours = []

    for fname, group in file_groups:
        group = group.sort_values('row_id')  # ensure temporal order
        idxs = group.index.tolist()
        start_idx = idxs[0]
        end_idx = idxs[-1] + 1

        file_indices.append((start_idx, end_idx))
        file_names.append(fname)

        # Site and hour from first window
        site_str = group['site'].iloc[0]
        hour = int(group['hour_utc'].iloc[0])
        file_sites.append(site_map.get(site_str, n_sites))  # n_sites = unknown
        file_hours.append(hour % 24)

        # Map labels
        for idx, rid in zip(idxs, group['row_id'].tolist()):
            if rid in label_map:
                all_labels[idx] = label_map[rid]
                all_mask[idx] = 1.0

    # Compute class frequencies for focal loss weighting
    labeled_labels = all_labels[all_mask > 0]
    class_freq = labeled_labels.sum(axis=0) + 1  # +1 smoothing
    class_weights = 1.0 / np.sqrt(class_freq)
    class_weights = class_weights / class_weights.mean()  # normalize to mean=1
    class_weights = np.clip(class_weights, 0.1, 10.0)

    print(f"Loaded {len(file_indices)} files, {int(all_mask.sum())} labeled windows, "
          f"{num_classes} classes, {n_sites} sites")
    print(f"Class weight range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")

    return {
        'embeddings': all_emb,
        'perch_logits': all_scores,
        'labels': all_labels,
        'mask': all_mask,
        'file_indices': file_indices,
        'file_names': file_names,
        'file_sites': np.array(file_sites),
        'file_hours': np.array(file_hours),
        'label_cols': label_cols,
        'class_weights': class_weights,
        'n_sites': n_sites,
        'num_classes': num_classes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_protossm(args, data):
    """Train ProtoSSM v5 with GroupKFold."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    n_files = len(data['file_indices'])
    file_names = np.array(data['file_names'])

    # GroupKFold: no leaking between files
    gkf = GroupKFold(n_splits=args.n_folds)
    groups = file_names

    all_oof_logits = np.zeros((len(data['embeddings']), data['num_classes']), dtype=np.float32)
    fold_models = []

    for fold, (train_fids, val_fids) in enumerate(gkf.split(range(n_files), groups=groups)):
        if args.fold is not None and fold != args.fold:
            continue

        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")

        # Create datasets
        train_ds = SoundscapeSequenceDataset(
            [data['file_indices'][i] for i in train_fids],
            data['embeddings'], data['perch_logits'], data['labels'], data['mask'],
            data['file_sites'][train_fids], data['file_hours'][train_fids],
            augment=True,
        )
        val_ds = SoundscapeSequenceDataset(
            [data['file_indices'][i] for i in val_fids],
            data['embeddings'], data['perch_logits'], data['labels'], data['mask'],
            data['file_sites'][val_fids], data['file_hours'][val_fids],
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=2, pin_memory=True)

        # Model
        model = ProtoSSMv5(
            emb_dim=1536,
            num_classes=data['num_classes'],
            d_model=args.d_model,
            d_state=args.d_state,
            n_ssm_layers=args.n_ssm_layers,
            n_heads=args.n_heads,
            n_prototypes=args.n_prototypes,
            n_sites=data['n_sites'],
            meta_dim=args.meta_dim,
            dropout=args.dropout,
            use_cross_attn=args.use_cross_attn,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"ProtoSSM params: {n_params:,}")

        # Loss
        loss_fn = SpeciesFocalLoss(
            gamma=args.focal_gamma,
            class_weights=data['class_weights'],
            label_smoothing=args.label_smoothing,
        )
        aux_loss_fn = nn.BCEWithLogitsLoss()

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
        # Cosine with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cosine_period, T_mult=1, eta_min=args.lr * 0.01
        )

        best_auc = 0.0
        best_state = None

        for epoch in range(args.epochs):
            # ── Train ──
            model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                # MixUp augmentation
                if args.mixup_alpha > 0 and np.random.random() < 0.5:
                    batch = embedding_mixup(batch, args.mixup_alpha)

                emb = batch['embeddings'].to(device)
                plogits = batch['perch_logits'].to(device)
                labels = batch['labels'].to(device)
                mask = batch['mask'].to(device)
                site = batch['site_id'].to(device)
                hour = batch['hour_id'].to(device)

                out = model(emb, plogits, site, hour)
                loss = loss_fn(out['logits'], labels, mask)

                # Auxiliary loss (no mask, regularizer)
                if args.aux_weight > 0:
                    aux_loss = aux_loss_fn(out['aux_logits'][mask.unsqueeze(-1).expand_as(out['aux_logits']) > 0],
                                           labels[mask.unsqueeze(-1).expand_as(labels) > 0])
                    loss = loss + args.aux_weight * aux_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step(epoch + n_batches / len(train_loader))

                train_loss += loss.item()
                n_batches += 1

            # ── Validate ──
            model.eval()
            all_preds, all_labels, all_masks = [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    emb = batch['embeddings'].to(device)
                    plogits = batch['perch_logits'].to(device)
                    site = batch['site_id'].to(device)
                    hour = batch['hour_id'].to(device)

                    out = model(emb, plogits, site, hour)
                    probs = torch.sigmoid(out['logits']).cpu().numpy()  # (B, T, C)

                    all_preds.append(probs)
                    all_labels.append(batch['labels'].numpy())
                    all_masks.append(batch['mask'].numpy())

            # Compute macro AUC on labeled windows only
            preds = np.concatenate(all_preds)   # (N_files, T, C)
            labels = np.concatenate(all_labels)
            masks = np.concatenate(all_masks)

            # Flatten: (N_files × T, C) then select labeled windows
            preds_flat = preds.reshape(-1, data['num_classes'])
            labels_flat = labels.reshape(-1, data['num_classes'])
            masks_flat = masks.reshape(-1)
            labeled_idx = masks_flat > 0

            if labeled_idx.sum() > 0:
                val_auc = macro_auc(labels_flat[labeled_idx], preds_flat[labeled_idx])
            else:
                val_auc = 0.0

            is_best = val_auc > best_auc
            if is_best:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={train_loss/n_batches:.4f} | "
                  f"val_auc={val_auc:.4f} | lr={lr_now:.2e}"
                  f"{' *BEST*' if is_best else ''}")

        # Save best model
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"protossm_fold{fold}.pt"
        torch.save({
            'model_state_dict': best_state,
            'fold': fold,
            'val_auc': best_auc,
            'config': {
                'd_model': args.d_model, 'd_state': args.d_state,
                'n_ssm_layers': args.n_ssm_layers, 'n_heads': args.n_heads,
                'n_prototypes': args.n_prototypes, 'meta_dim': args.meta_dim,
                'n_sites': data['n_sites'], 'num_classes': data['num_classes'],
                'use_cross_attn': args.use_cross_attn,
            },
        }, ckpt_path)
        print(f"Saved {ckpt_path} (val_auc={best_auc:.4f})")

        # Generate OOF predictions for this fold
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                emb = batch['embeddings'].to(device)
                plogits = batch['perch_logits'].to(device)
                site = batch['site_id'].to(device)
                hour = batch['hour_id'].to(device)

                out = model(emb, plogits, site, hour)
                logits = out['logits'].cpu().numpy()  # (B, T, C)

                # Map back to original indices
                for i in range(logits.shape[0]):
                    file_idx = val_fids[batch_idx * args.batch_size * 2 + i]
                    start, end = data['file_indices'][file_idx]
                    T = min(end - start, 12)
                    all_oof_logits[start:start+T] = logits[i, :T]

        fold_models.append({'fold': fold, 'val_auc': best_auc})

    return all_oof_logits, fold_models


def train_resboosting(args, data, oof_logits):
    """Train ResidualSSM on first-pass residuals."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("RESBOOSTING (ResidualSSM)")
    print(f"{'='*60}")

    # Compute residuals
    labeled_mask = data['mask'] > 0
    oof_probs = 1.0 / (1.0 + np.exp(-oof_logits))  # sigmoid
    residuals = data['labels'] - oof_probs  # in [-1, 1]

    print(f"Residual stats: mean={residuals[labeled_mask].mean():.4f}, "
          f"std={residuals[labeled_mask].std():.4f}")

    n_files = len(data['file_indices'])
    file_names = np.array(data['file_names'])
    gkf = GroupKFold(n_splits=args.n_folds)

    best_correction_weight = 0.0
    best_correction_auc = 0.0
    all_corrections = np.zeros_like(oof_logits)

    for fold, (train_fids, val_fids) in enumerate(gkf.split(range(n_files), groups=file_names)):
        if args.fold is not None and fold != args.fold:
            continue

        print(f"\n  Fold {fold}")

        # Create datasets
        train_ds = SoundscapeSequenceDataset(
            [data['file_indices'][i] for i in train_fids],
            data['embeddings'], oof_logits, residuals, data['mask'],
            data['file_sites'][train_fids], data['file_hours'][train_fids],
        )
        val_ds = SoundscapeSequenceDataset(
            [data['file_indices'][i] for i in val_fids],
            data['embeddings'], oof_logits, residuals, data['mask'],
            data['file_sites'][val_fids], data['file_hours'][val_fids],
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=2)

        model = ResidualSSM(
            emb_dim=1536,
            num_classes=data['num_classes'],
            d_model=128,
            d_state=16,
            n_ssm_layers=2,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.res_epochs)

        best_loss = float('inf')
        best_state = None

        for epoch in range(args.res_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                emb = batch['embeddings'].to(device)
                first_pass = batch['perch_logits'].to(device)  # these are oof_logits
                target_residuals = batch['labels'].to(device)   # these are residuals
                mask = batch['mask'].to(device)

                corrections = model(emb, first_pass)

                # MSE loss on residuals, masked
                mse = ((corrections - target_residuals) ** 2)
                if mask is not None:
                    mask_expand = mask.unsqueeze(-1)
                    mse = (mse * mask_expand).sum() / (mask_expand.sum() * mse.shape[-1] + 1e-8)
                else:
                    mse = mse.mean()

                optimizer.zero_grad()
                mse.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += mse.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{args.res_epochs} | mse={avg_loss:.6f}")

        # Save
        out_dir = Path(args.output_dir)
        torch.save({
            'model_state_dict': best_state,
            'fold': fold,
        }, out_dir / f"resssm_fold{fold}.pt")

        # Generate corrections for this fold's validation
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                emb = batch['embeddings'].to(device)
                first_pass = batch['perch_logits'].to(device)
                corrections = model(emb, first_pass).cpu().numpy()

                for i in range(corrections.shape[0]):
                    file_idx = val_fids[batch_idx * args.batch_size * 2 + i]
                    start, end = data['file_indices'][file_idx]
                    T = min(end - start, 12)
                    all_corrections[start:start+T] = corrections[i, :T]

    # Grid search correction weight
    print("\n  Grid search correction weight:")
    labeled_idx = data['mask'] > 0
    for w in np.arange(0.0, 0.55, 0.05):
        corrected_logits = oof_logits + w * all_corrections
        corrected_probs = 1.0 / (1.0 + np.exp(-corrected_logits))
        auc = macro_auc(data['labels'][labeled_idx], corrected_probs[labeled_idx])
        marker = " *" if auc > best_correction_auc else ""
        if auc > best_correction_auc:
            best_correction_auc = auc
            best_correction_weight = w
        print(f"    w={w:.2f} | auc={auc:.4f}{marker}")

    print(f"\n  Best correction weight: {best_correction_weight:.2f} (auc={best_correction_auc:.4f})")

    # Save config
    with open(Path(args.output_dir) / 'resboost_config.json', 'w') as f:
        json.dump({'correction_weight': float(best_correction_weight)}, f)

    return best_correction_weight


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def macro_auc(y_true, y_pred):
    """Competition metric: macro-averaged ROC-AUC, skipping classes with no positives."""
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == col.shape[0]:
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return np.mean(aucs) if aucs else 0.0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--perch_npz', type=str, default='kaggle_model/all_perch_arrays.npz')
    parser.add_argument('--perch_meta', type=str, default='kaggle_model/all_perch_meta.parquet')

    # ProtoSSM architecture
    parser.add_argument('--d_model', type=int, default=320)
    parser.add_argument('--d_state', type=int, default=32)
    parser.add_argument('--n_ssm_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_prototypes', type=int, default=2)
    parser.add_argument('--meta_dim', type=int, default=24)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_cross_attn', action='store_true', default=True)

    # Training
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--focal_gamma', type=float, default=2.5)
    parser.add_argument('--label_smoothing', type=float, default=0.03)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    parser.add_argument('--aux_weight', type=float, default=0.1)
    parser.add_argument('--cosine_period', type=int, default=20)

    # ResBoosting
    parser.add_argument('--res_epochs', type=int, default=40)

    # CV
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=None, help='Train single fold (None=all)')
    parser.add_argument('--seed', type=int, default=42)

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/protossm_v5')

    args = parser.parse_args()
    set_seed(args.seed)

    print("Loading data...")
    data = load_data(args)

    print("\n" + "="*60)
    print("PHASE 1: ProtoSSM v5 Training")
    print("="*60)
    oof_logits, fold_models = train_protossm(args, data)

    # Evaluate OOF
    labeled_idx = data['mask'] > 0
    oof_probs = 1.0 / (1.0 + np.exp(-oof_logits))
    oof_auc = macro_auc(data['labels'][labeled_idx], oof_probs[labeled_idx])
    print(f"\nOOF macro AUC (ProtoSSM): {oof_auc:.4f}")

    print("\n" + "="*60)
    print("PHASE 2: ResBoosting")
    print("="*60)
    correction_weight = train_resboosting(args, data, oof_logits)

    print(f"\n{'='*60}")
    print(f"DONE — models saved to {args.output_dir}/")
    print(f"ProtoSSM OOF AUC: {oof_auc:.4f}")
    print(f"Best correction weight: {correction_weight:.2f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
