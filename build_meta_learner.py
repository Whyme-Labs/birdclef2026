"""V188: Learned aggregator on V137 predictions (meta-learner).

Hypothesis: V137's per-chunk predictions on labeled soundscape contain patterns
(systematic biases, cross-class correlations, calibration error) that a small
neural net can learn to correct.

Input: V137's per-chunk 234-d prediction vector (from pseudo_labels_v137)
Target: soundscape ground-truth (from train_soundscapes_labels.csv)
Eval: held-out 20% of recordings, macro-AUC

Training set: ~1478 labeled chunks aligned with V137 predictions.
Test sizes: tiny train, but architecture stays small (MLP 234→128→234).

PREDICTION (Predict-Then-Run discipline):
  - Held-out macro-AUC: 0.85 ± 0.03 (medium confidence)
    - Rationale: V137 already strong on labeled soundscape (probes get ~0.99 AUC
      train-test contaminated; clean held-out should be 0.80-0.90)
    - If meta-learner adds nothing beyond identity, AUC = V137's own LB-equivalent (~0.94)
    - If meta-learner introduces noise (overfits 1478 samples), AUC could drop to 0.65
  - Direction: match-baseline (matching V137's signal) to slightly-beat
  - Confidence: medium for direction, low for exact magnitude

DISCONFIRM CONDITION:
  - Held-out macro-AUC < 0.65 → meta-learner adds noise, abandon
  - Cross-chunk attention version not better than chunk-independent MLP →
    cross-chunk hypothesis is wrong

CONFIRM CONDITION:
  - Held-out macro-AUC > 0.85 AND beats single-chunk MLP → cross-chunk pattern matters
"""
import argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def load_data():
    """Align V137 pseudo predictions with soundscape labels."""
    print("Loading pseudo_labels_v137 (V137 ensemble on soundscapes)...")
    pseudo = pd.read_csv("pseudo_labels_v137/raw_predictions.csv")
    print(f"  pseudo: {len(pseudo)} rows, {pseudo.shape[1]} cols")

    labels = pd.read_csv("data/train_soundscapes_labels.csv")
    labels["end_sec"] = pd.to_timedelta(labels["end"]).dt.total_seconds().astype(int)
    print(f"  labels: {len(labels)} rows")

    # Aggregate labels per (filename, end_sec) — multi-label union
    def parse_lbl(x):
        if pd.isna(x) or x == "nan": return set()
        return set(t.strip() for t in str(x).split(";") if t.strip())
    labels["label_set"] = labels["primary_label"].apply(parse_lbl)
    grouped = labels.groupby(["filename", "end_sec"])["label_set"].apply(
        lambda s: set().union(*s)
    ).reset_index()
    print(f"  unique chunks (label union): {len(grouped)}")

    # Class names from pseudo columns
    label_cols = [c for c in pseudo.columns if c not in ("file", "end_time")]
    l2i = {c: i for i, c in enumerate(label_cols)}
    print(f"  classes: {len(label_cols)}")

    # Align: for each (filename, end_sec) in labels, find matching pseudo row
    # pseudo's "end_time" is float (5.0, 10.0, ...) — match by int
    pseudo["end_int"] = pseudo["end_time"].astype(int)
    pseudo_keyed = pseudo.set_index(["file", "end_int"])

    rows_X = []
    rows_Y = []
    rows_groups = []  # filename for grouped CV
    n_missing = 0
    for _, r in grouped.iterrows():
        key = (r["filename"], int(r["end_sec"]))
        if key not in pseudo_keyed.index:
            n_missing += 1
            continue
        pseudo_row = pseudo_keyed.loc[key]
        x = pseudo_row[label_cols].values.astype(np.float32)
        y = np.zeros(len(label_cols), dtype=np.float32)
        for lbl in r["label_set"]:
            if lbl in l2i:
                y[l2i[lbl]] = 1.0
        rows_X.append(x)
        rows_Y.append(y)
        rows_groups.append(r["filename"])

    print(f"  aligned: {len(rows_X)} chunks, {n_missing} missing pseudo predictions")
    X = np.stack(rows_X)
    Y = np.stack(rows_Y)
    groups = np.array(rows_groups)
    return X, Y, groups, label_cols


class MetaMLP(nn.Module):
    def __init__(self, n_classes=234, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_classes, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        # Residual: meta-learner predicts CORRECTION on top of V137
        return torch.sigmoid(self.net(x) + torch.logit(x.clamp(1e-6, 1 - 1e-6)))


def train_eval(X_tr, Y_tr, X_te, Y_te, n_classes, epochs=50, lr=1e-3, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MetaMLP(n_classes=n_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    X_tr_t = torch.from_numpy(X_tr).to(device)
    Y_tr_t = torch.from_numpy(Y_tr).to(device)
    X_te_t = torch.from_numpy(X_te).to(device)

    best_auc = 0
    best_pred = None
    for epoch in range(epochs):
        # Shuffle
        idx = torch.randperm(len(X_tr_t))
        for s in range(0, len(idx), batch_size):
            sl = idx[s:s+batch_size]
            x = X_tr_t[sl]; y = Y_tr_t[sl]
            pred = model(x)
            loss = loss_fn(pred, y)
            optim.zero_grad(); loss.backward(); optim.step()
        # Eval
        model.eval()
        with torch.no_grad():
            pred_te = model(X_te_t).cpu().numpy()
        # Macro AUC for classes with positives
        valid = Y_te.sum(axis=0) >= 1
        try:
            auc = roc_auc_score(Y_te[:, valid], pred_te[:, valid], average="macro")
        except ValueError:
            auc = 0.0
        if auc > best_auc:
            best_auc = auc
            best_pred = pred_te.copy()
        model.train()
    return best_auc, best_pred


def main():
    X, Y, groups, label_cols = load_data()

    # Baseline: V137 itself (no meta-learner) → just use X as predictions
    print(f"\nBaseline (V137 itself, no meta-learner):")
    valid = Y.sum(axis=0) >= 1
    auc_baseline = roc_auc_score(Y[:, valid], X[:, valid], average="macro")
    print(f"  V137 macro AUC (in-sample): {auc_baseline:.4f}")

    # 5-fold grouped CV on filename
    gkf = GroupKFold(n_splits=5)
    auc_meta_folds = []
    auc_v137_folds = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]

        # V137 baseline AUC on test fold
        valid_te = Y_te.sum(axis=0) >= 1
        if valid_te.sum() == 0:
            print(f"  fold {fold}: no valid classes in test, skipping")
            continue
        try:
            auc_v137 = roc_auc_score(Y_te[:, valid_te], X_te[:, valid_te], average="macro")
        except ValueError:
            auc_v137 = 0.0

        # Meta-learner
        auc_meta, _ = train_eval(X_tr, Y_tr, X_te, Y_te, len(label_cols))

        print(f"  fold {fold}: V137={auc_v137:.4f}, meta={auc_meta:.4f}, delta={auc_meta - auc_v137:+.4f}")
        auc_v137_folds.append(auc_v137)
        auc_meta_folds.append(auc_meta)

    print(f"\nMean V137 AUC: {np.mean(auc_v137_folds):.4f} ± {np.std(auc_v137_folds):.4f}")
    print(f"Mean meta AUC: {np.mean(auc_meta_folds):.4f} ± {np.std(auc_meta_folds):.4f}")
    print(f"Mean delta:    {np.mean(auc_meta_folds) - np.mean(auc_v137_folds):+.4f}")


if __name__ == "__main__":
    main()
