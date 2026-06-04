"""
PCXT — Perch Cross-Window Transformer.

Takes Perch v2's 1536-d embeddings (per 5s window) and applies a small transformer
across the 12 windows of a 60s file before predicting the 234 BirdCLEF species.
This sidesteps Perch's own classifier (trained on isolated Xeno-Canto recordings)
and replaces V137's per-class linear probe with a single jointly-trained model
that has cross-window awareness.

Why this is meaningfully different from previous attempts:
  • V137 perch_probs: independent per-class linear probe on per-window embedding.
    No cross-window reasoning.
  • Path 2 (V154): cross-window attention on EffNetV2-S backbone features. Backbone
    was warm-started from R3 → predictions correlated with V2S, no orthogonal
    signal in the ensemble.
  • PCXT: cross-window attention on PERCH embeddings (Google's 14k-class pretraining
    on millions of bird recordings, much richer per-window features than EffNetV2-S
    learned from BirdCLEF data alone). Different feature space, different bias.

Training data:
  • Cached Perch embeddings on 432 training soundscape files (5184 windows).
  • Hard labels for 66 of those files from `train_soundscapes_labels.csv`.
  • Soft pseudo-labels for the rest from R4 `raw_predictions.csv`.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class PCXT(nn.Module):
    def __init__(self, emb_dim=1536, n_windows=12, num_classes=234,
                 d_inner=256, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.proj_in = nn.Linear(emb_dim, d_inner)
        self.pos_emb = nn.Parameter(torch.randn(1, n_windows, d_inner) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_inner, nhead=n_heads,
            dim_feedforward=d_inner * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_inner)
        self.head = nn.Linear(d_inner, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, emb):  # (B, W, 1536)
        h = self.proj_in(emb) + self.pos_emb
        h = self.encoder(h)
        h = self.norm(h)
        h = self.drop(h)
        return self.head(h)  # (B, W, num_classes) — raw logits


def parse_time_str(t):
    parts = str(t).split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def build_label_matrix(meta_df, hard_csv, soft_csv, label_cols, n_windows=12):
    """
    Returns (Y_hard, Y_soft, has_hard_mask) of shape (n_files, n_windows, n_classes).
    has_hard_mask[i] = True if file i has any hand label coverage.
    """
    l2i = {c: i for i, c in enumerate(label_cols)}
    files = meta_df["filename"].drop_duplicates().tolist()
    n_files = len(files)
    f2i = {f: i for i, f in enumerate(files)}
    Y_hard = np.zeros((n_files, n_windows, len(label_cols)), dtype=np.float32)
    Y_soft = np.zeros((n_files, n_windows, len(label_cols)), dtype=np.float32)
    has_hard = np.zeros(n_files, dtype=bool)

    if hard_csv and Path(hard_csv).exists():
        sc = pd.read_csv(hard_csv).drop_duplicates()
        sc["primary_label"] = sc["primary_label"].astype(str)
        sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
        for _, row in sc.iterrows():
            fn = row["filename"]
            if fn not in f2i:
                continue
            wi = (row["end_sec"] // 5) - 1
            if not (0 <= wi < n_windows):
                continue
            for lbl in str(row["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in l2i:
                    Y_hard[f2i[fn], wi, l2i[lbl]] = 1.0
            has_hard[f2i[fn]] = True

    if soft_csv and Path(soft_csv).exists():
        ps = pd.read_csv(soft_csv)
        # Soft pseudo-label CSV columns: file, end_time, c0, c1, …
        # End_time is e.g. 5.0, 10.0, …, 60.0 → window index = end_time/5 - 1
        for _, row in ps.iterrows():
            fn = row["file"]
            if fn not in f2i:
                continue
            wi = int(row["end_time"]) // 5 - 1
            if not (0 <= wi < n_windows):
                continue
            for ci, col in enumerate(label_cols):
                if col in ps.columns:
                    v = float(row[col])
                    Y_soft[f2i[fn], wi, ci] = v

    return Y_hard, Y_soft, has_hard, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npz", default="kaggle_model/all_perch_arrays.npz")
    ap.add_argument("--meta_parquet", default="kaggle_model/all_perch_meta.parquet")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--soundscape_labels", default="data/train_soundscapes_labels.csv")
    ap.add_argument("--soft_csv", default="pseudo_labels_r4/raw_predictions.csv")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--d_inner", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--soft_weight", type=float, default=0.3,
                    help="Weight on soft pseudo-label component of loss (vs hard).")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_pt", default="kaggle_model/pcxt.pt")
    ap.add_argument("--out_onnx", default="kaggle_model/pcxt.onnx")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Labels
    tax = pd.read_csv(args.taxonomy)
    label_cols = sorted(tax["primary_label"].astype(str).tolist())
    N_CLASSES = len(label_cols)

    # Cached Perch embeddings + meta
    d = np.load(args.emb_npz)
    emb_full = d["emb_full"]
    meta = pd.read_parquet(args.meta_parquet)
    n_files = meta["filename"].nunique()
    n_windows = 12
    assert emb_full.shape[0] == n_files * n_windows, (
        f"emb_full rows ({emb_full.shape[0]}) != n_files*n_windows ({n_files*n_windows})"
    )
    print(f"Loaded {emb_full.shape} Perch embeddings, {n_files} files, {n_windows} windows.")

    # Reshape embeddings to (n_files, n_windows, 1536)
    X = emb_full.reshape(n_files, n_windows, -1).astype(np.float32)

    # Build labels
    Y_hard, Y_soft, has_hard, files = build_label_matrix(
        meta.drop_duplicates("filename"), args.soundscape_labels, args.soft_csv,
        label_cols, n_windows=n_windows,
    )
    print(f"Labels: hard files={has_hard.sum()}, soft files={n_files}, total {n_files} files.")
    print(f"Hard positives per window (avg): {Y_hard.sum(axis=-1).mean():.2f}")
    print(f"Soft positives per window (>0.5 avg): {(Y_soft > 0.5).sum(axis=-1).mean():.2f}")

    # File-level split: hand-labeled files into train (most) + val (some)
    rng = np.random.default_rng(args.seed)
    hard_idx = np.where(has_hard)[0]
    soft_only_idx = np.where(~has_hard)[0]
    perm = rng.permutation(hard_idx)
    n_val = max(8, int(len(hard_idx) * args.val_frac))
    val_idx = perm[:n_val]
    train_hard_idx = perm[n_val:]
    train_idx = np.concatenate([train_hard_idx, soft_only_idx])
    print(f"Split: {len(train_idx)} train ({len(train_hard_idx)} hard + {len(soft_only_idx)} soft), {n_val} val (all hard)")

    Xt = torch.from_numpy(X[train_idx]).to(device)
    Yh_t = torch.from_numpy(Y_hard[train_idx]).to(device)
    Ys_t = torch.from_numpy(Y_soft[train_idx]).to(device)
    has_hard_t = torch.from_numpy(has_hard[train_idx]).to(device)
    Xv = torch.from_numpy(X[val_idx]).to(device)
    Yh_v = torch.from_numpy(Y_hard[val_idx]).to(device)

    model = PCXT(
        emb_dim=X.shape[-1], n_windows=n_windows, num_classes=N_CLASSES,
        d_inner=args.d_inner, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"PCXT: {n_params:.0f}k params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = 0.0
    best_state = None
    for ep in range(args.epochs):
        model.train()
        # Mini-batch over files
        perm_b = torch.randperm(len(Xt), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, len(perm_b), args.batch_size):
            idx = perm_b[s:s + args.batch_size]
            xb = Xt[idx]
            yh = Yh_t[idx]
            ys = Ys_t[idx]
            mask_hard = has_hard_t[idx].view(-1, 1, 1)  # (B, 1, 1)

            opt.zero_grad()
            logits = model(xb)  # (B, W, C)

            # Hard component on hard-labeled files
            if mask_hard.any():
                hl = F.binary_cross_entropy_with_logits(
                    logits, yh, reduction="none"
                )
                hl = (hl * mask_hard.float()).sum() / (mask_hard.float().sum() * yh.shape[-1] * yh.shape[-2] + 1e-9)
            else:
                hl = torch.zeros((), device=device)

            # Soft component: use soft labels only on soft-only files (to avoid double signal)
            soft_mask = (~has_hard_t[idx]).view(-1, 1, 1).float()
            if soft_mask.any():
                # Use soft labels with binary CE on probabilities
                sl = F.binary_cross_entropy_with_logits(
                    logits, ys, reduction="none"
                )
                sl = (sl * soft_mask).sum() / (soft_mask.sum() * ys.shape[-1] * ys.shape[-2] + 1e-9)
            else:
                sl = torch.zeros((), device=device)

            loss = hl + args.soft_weight * sl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()

        # Validate on hard-labeled val
        model.eval()
        with torch.no_grad():
            v_logits = model(Xv)
            v_probs = torch.sigmoid(v_logits).cpu().numpy()
        Y_v = Yh_v.cpu().numpy()
        # Macro AUC over species with both pos and neg
        flat_p = v_probs.reshape(-1, N_CLASSES)
        flat_y = Y_v.reshape(-1, N_CLASSES)
        aucs = []
        for c in range(N_CLASSES):
            yc = flat_y[:, c]
            if 1 <= yc.sum() < len(yc):
                try: aucs.append(roc_auc_score(yc, flat_p[:, c]))
                except Exception: pass
        val_auc = float(np.mean(aucs)) if aucs else 0.0

        if (ep + 1) % 5 == 0 or ep < 5:
            print(f"ep{ep+1:02d} loss={epoch_loss / max(n_batches, 1):.4f} val_auc={val_auc:.4f} (n_aucs={len(aucs)})")

        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest val AUC: {best_val:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save weights + ONNX
    Path(args.out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "config": {"emb_dim": X.shape[-1], "n_windows": n_windows, "num_classes": N_CLASSES,
                   "d_inner": args.d_inner, "n_heads": args.n_heads, "n_layers": args.n_layers,
                   "dropout": args.dropout},
        "label_cols": label_cols,
        "best_val_auc": best_val,
    }, args.out_pt)
    print(f"Saved weights to {args.out_pt}")

    # ONNX export with sigmoid baked in
    class Wrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return torch.sigmoid(self.m(x))
    wrapped = Wrap(model).eval()
    dummy = torch.zeros(1, n_windows, X.shape[-1], device=device)
    torch.onnx.export(
        wrapped, dummy, args.out_onnx,
        input_names=["emb"], output_names=["probs"],
        dynamic_axes={"emb": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX to {args.out_onnx}")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out_onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    test = rng.standard_normal((2, n_windows, X.shape[-1])).astype(np.float32)
    out_onnx = sess.run(None, {inp: test})[0]
    with torch.no_grad():
        out_torch = wrapped(torch.from_numpy(test).to(device)).cpu().numpy()
    diff = np.abs(out_onnx - out_torch).max()
    print(f"ONNX vs PyTorch max diff: {diff:.2e}")


if __name__ == "__main__":
    main()
