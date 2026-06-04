"""Probe orthogonality + optimal-combination analysis.

Question 1: Do our SSL probes carry the SAME signal (redundant) or DIFFERENT
signal (orthogonal)? Measured via pairwise correlation of held-out predictions.

Question 2: What's the OPTIMAL way to combine them? Tests:
  - Naive averaging at K probes (K=1..5)
  - Learned weighted average per probe
  - Per-class learned weights (different classes prefer different probes)

Inputs (all on the same 739 labeled-soundscape chunks):
  - kaggle_model/birdmae_train_emb.npz: (739, 768)
  - kaggle_model/biolingual_train_emb.npz: (739, 512)
  - kaggle_model/birdaves_train_emb.npz: (739, 768)
  - cache/mimo_v190_features.npz: (739, 1280)

For each probe: train per-class LR via 5-fold GroupKFold; get OOF predictions.
Then compute correlations + ensemble AUCs.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


PROBE_FILES = {
    "BirdMAE": "kaggle_model/birdmae_train_emb.npz",
    "AudioMAE": "kaggle_model/audiomae_train_emb.npz",
    "BioLingual": "kaggle_model/biolingual_train_emb.npz",
    "BirdAVES": "kaggle_model/birdaves_train_emb.npz",
    "MiMo": "cache/mimo_v190_features.npz",
    "WavJEPA": "kaggle_model/wavjepa_train_emb.npz",
}


def load_probe(path):
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    Y = z["Y"].astype(np.float32)
    if "groups" in z.files:
        g = z["groups"]
    else:
        g = None
    return X, Y, g


def get_oof_predictions(X, Y, groups, label_cols, n_splits=5, min_pos=3):
    """5-fold GroupKFold per-class LR; return OOF (N, 234) predictions
    and per-fold valid masks unioned to per-class boolean."""
    gkf = GroupKFold(n_splits=n_splits)
    n = X.shape[0]
    n_classes = Y.shape[1]
    oof = np.zeros((n, n_classes), dtype=np.float32)
    valid_classes = np.zeros(n_classes, dtype=bool)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]
        # Standardize within fold
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-6
        X_tr_n = (X_tr - mu) / sd
        X_te_n = (X_te - mu) / sd
        for ci in range(n_classes):
            if Y_tr[:, ci].sum() < min_pos:
                continue
            try:
                lr = LogisticRegression(C=1.0, max_iter=300,
                                         solver="liblinear",
                                         class_weight="balanced")
                lr.fit(X_tr_n, Y_tr[:, ci])
                oof[te_idx, ci] = lr.predict_proba(X_te_n)[:, 1]
                valid_classes[ci] = True
            except Exception:
                pass
    return oof, valid_classes


def macro_auc(preds, Y, valid_mask):
    """Per-class AUC averaged over classes with both train+test positives."""
    aucs = []
    for ci in np.where(valid_mask)[0]:
        if Y[:, ci].sum() < 1:
            continue
        try:
            aucs.append(roc_auc_score(Y[:, ci], preds[:, ci]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else 0.0, len(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundscape_csv", default="data/train_soundscapes_labels.csv")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    args = ap.parse_args()

    tax = pd.read_csv(args.taxonomy)
    label_cols = tax["primary_label"].astype(str).tolist()

    # Load groups from soundscape CSV (need filename per chunk for GroupKFold)
    sc = pd.read_csv(args.soundscape_csv)
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)

    def parse_lbl(x):
        if pd.isna(x) or x == "nan":
            return set()
        return set(t.strip() for t in str(x).split(";") if t.strip())

    sc["label_set"] = sc["primary_label"].apply(parse_lbl)
    grouped = sc.groupby(["filename", "end_sec"])["label_set"].apply(
        lambda s: set().union(*s)).reset_index()
    audio_dir = Path("data/train_soundscapes")
    existing = set(p.name for p in audio_dir.glob("*.ogg"))
    grouped = grouped[grouped["filename"].isin(existing)].reset_index(drop=True)
    groups_canonical = grouped["filename"].values
    Y_canonical = np.stack([
        [1.0 if c in lset else 0.0 for c in label_cols]
        for lset in grouped["label_set"]
    ]).astype(np.float32)
    print(f"Canonical chunks: {len(grouped)}, classes: {len(label_cols)}")

    # Load each probe and verify alignment
    probe_data = {}
    for name, path in PROBE_FILES.items():
        if not Path(path).exists():
            print(f"  SKIP {name}: {path} missing")
            continue
        X, Y_p, g = load_probe(path)
        # Y should match canonical (same chunk order); use canonical to be safe
        if Y_p.shape != Y_canonical.shape or X.shape[0] != len(grouped):
            print(f"  SKIP {name}: shape mismatch X={X.shape}, Y={Y_p.shape}")
            continue
        probe_data[name] = X
        print(f"  {name}: X={X.shape}")

    if not probe_data:
        print("No probes loaded; abort")
        return

    # Get OOF predictions per probe
    print("\nComputing 5-fold GroupKFold OOF predictions per probe...")
    oof_preds = {}
    aucs = {}
    valid_masks = {}
    for name, X in probe_data.items():
        oof, valid = get_oof_predictions(X, Y_canonical, groups_canonical, label_cols)
        auc, n_classes = macro_auc(oof, Y_canonical, valid)
        oof_preds[name] = oof
        aucs[name] = auc
        valid_masks[name] = valid
        print(f"  {name}: AUC={auc:.4f} on {n_classes} classes")

    # ============================================================
    # Question 1: Pairwise correlation between probe predictions
    # ============================================================
    print(f"\n{'='*60}")
    print("Q1: Pairwise correlation of probe predictions (averaged across classes)")
    print(f"{'='*60}")
    names = list(oof_preds.keys())
    n_probes = len(names)
    # For each class, compute pairwise correlation of predictions; average
    per_pair_corr = np.zeros((n_probes, n_probes))
    common_valid = np.ones(Y_canonical.shape[1], dtype=bool)
    for vm in valid_masks.values():
        common_valid &= vm
    common_valid &= (Y_canonical.sum(axis=0) >= 1)
    print(f"Classes with valid predictions in ALL probes AND ≥1 positive: {common_valid.sum()}")
    valid_idx = np.where(common_valid)[0]
    for i in range(n_probes):
        for j in range(n_probes):
            if i == j:
                per_pair_corr[i, j] = 1.0
                continue
            corrs = []
            for ci in valid_idx:
                p_i = oof_preds[names[i]][:, ci]
                p_j = oof_preds[names[j]][:, ci]
                if p_i.std() < 1e-6 or p_j.std() < 1e-6:
                    continue
                c = np.corrcoef(p_i, p_j)[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
            per_pair_corr[i, j] = np.mean(corrs) if corrs else 0.0

    print("\nMean per-class Pearson correlation matrix:")
    header = "          " + "  ".join(f"{n:>10s}" for n in names)
    print(header)
    for i, n in enumerate(names):
        row = f"{n:>10s}  " + "  ".join(f"{per_pair_corr[i, j]:>10.4f}" for j in range(n_probes))
        print(row)

    # ============================================================
    # Question 2: AUC trajectory under naive averaging at K probes
    # ============================================================
    print(f"\n{'='*60}")
    print("Q2: AUC under naive averaging of K probes")
    print(f"{'='*60}")
    from itertools import combinations
    best_at_k = []
    for k in range(1, n_probes + 1):
        best_auc = 0.0
        best_combo = None
        for combo in combinations(names, k):
            stacked = np.mean([oof_preds[n] for n in combo], axis=0)
            cm = np.ones(Y_canonical.shape[1], dtype=bool)
            for n in combo:
                cm &= valid_masks[n]
            cm &= (Y_canonical.sum(axis=0) >= 1)
            auc, _ = macro_auc(stacked, Y_canonical, cm)
            if auc > best_auc:
                best_auc = auc
                best_combo = combo
        best_at_k.append((k, best_auc, best_combo))
        print(f"  k={k}  best AUC={best_auc:.4f}  combo={best_combo}")

    # ============================================================
    # Question 3: Learned probe weights (per-class)
    # ============================================================
    print(f"\n{'='*60}")
    print("Q3: Learned weighted average (single weight per probe across classes)")
    print(f"{'='*60}")
    # Stack OOF preds as input features: shape (N, n_probes), per class
    # Find optimal weights via simple grid search on class-averaged AUC
    base = np.stack([oof_preds[n] for n in names])  # (n_probes, N, C)
    common_valid = np.ones(Y_canonical.shape[1], dtype=bool)
    for vm in valid_masks.values():
        common_valid &= vm
    common_valid &= (Y_canonical.sum(axis=0) >= 1)
    # Simplex grid w/ step 0.1 over probes
    best_w_auc = 0.0
    best_w = None
    if n_probes <= 5:
        from itertools import product
        step = 0.2
        steps = np.arange(0, 1.01, step)
        for ws in product(steps, repeat=n_probes):
            s = sum(ws)
            if abs(s - 1.0) > 1e-6:
                continue
            stacked = np.tensordot(np.array(ws), base, axes=([0], [0]))
            auc, _ = macro_auc(stacked, Y_canonical, common_valid)
            if auc > best_w_auc:
                best_w_auc = auc
                best_w = ws
        print(f"  Best uniform-grid weights (step={step}): "
              f"{dict(zip(names, [f'{w:.2f}' for w in best_w]))}")
        print(f"  Best weighted AUC: {best_w_auc:.4f}")
    else:
        print("  (grid search skipped for n_probes > 5; equal averaging only)")
    eq = np.mean(base, axis=0)
    eq_auc, _ = macro_auc(eq, Y_canonical, common_valid)
    print(f"  Equal averaging AUC: {eq_auc:.4f}")
    if best_w is not None:
        print(f"  Delta from learned: {best_w_auc - eq_auc:+.4f}")

    # ============================================================
    # Question 4: How much of orthogonal info is being captured?
    # ============================================================
    print(f"\n{'='*60}")
    print("Q4: Diversity quantification")
    print(f"{'='*60}")
    # Off-diagonal mean of correlation matrix
    od_mask = ~np.eye(n_probes, dtype=bool)
    mean_corr = float(per_pair_corr[od_mask].mean())
    max_corr = float(per_pair_corr[od_mask].max())
    min_corr = float(per_pair_corr[od_mask].min())
    print(f"  Off-diagonal correlation: mean={mean_corr:.3f}, "
          f"min={min_corr:.3f}, max={max_corr:.3f}")
    # Compound index: ensemble AUC > best single iff probes are orthogonal
    best_single = max(aucs.values())
    best_eq_avg = best_at_k[-1][1]
    delta = best_eq_avg - best_single
    print(f"  Best single probe AUC: {best_single:.4f}")
    print(f"  Best equal-avg AUC: {best_eq_avg:.4f}")
    print(f"  Compound delta (avg - best_single): {delta:+.4f}")
    print()
    if mean_corr > 0.85:
        print(f"  VERDICT: probes are highly redundant (mean corr {mean_corr:.2f}). "
              "Adding more from same family unlikely to compound.")
    elif mean_corr > 0.65:
        print(f"  VERDICT: probes are moderately correlated (mean corr {mean_corr:.2f}). "
              "Some compound but diminishing returns.")
    else:
        print(f"  VERDICT: probes are diverse (mean corr {mean_corr:.2f}). "
              "More probes from this family should still compound.")


if __name__ == "__main__":
    main()
