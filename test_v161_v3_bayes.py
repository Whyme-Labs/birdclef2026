"""V161-v3: soundscape-level Bayesian filtering — boost present species, suppress absent."""
import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(labels, preds):
    aucs = []
    for j in range(labels.shape[1]):
        if labels[:, j].sum() > 0 and labels[:, j].sum() < len(labels[:, j]):
            try:
                aucs.append(roc_auc_score(labels[:, j], preds[:, j]))
            except ValueError:
                pass
    return np.mean(aucs), len(aucs)


def soundscape_bayes_filter(preds_3d, K_top, gamma, mode="zscore"):
    """
    preds_3d: (n_files, 12, 234)
    For each soundscape, compute top-K-mean per species → presence score.
    Z-score (or rank) presence across soundscapes, then multiply chunk preds by
    factor f = 1 + γ * tanh(z) so evident species get boosted, absent suppressed.
    Returns same shape.
    """
    # Per-soundscape, per-species presence (top-K mean across chunks)
    sorted_preds = np.sort(preds_3d, axis=1)
    top_K = sorted_preds[:, -K_top:, :].mean(axis=1)  # (n_files, 234)

    if mode == "zscore":
        # Z-score across soundscapes per species
        mu = top_K.mean(axis=0, keepdims=True)
        sd = top_K.std(axis=0, keepdims=True) + 1e-9
        z = (top_K - mu) / sd  # (n_files, 234)
        factor = 1.0 + gamma * np.tanh(z)
    elif mode == "rank":
        # Per-species rank across soundscapes
        ranks = np.argsort(np.argsort(top_K, axis=0), axis=0).astype(np.float32) / max(1, top_K.shape[0] - 1)
        # Center rank to [-1, 1]
        z = 2.0 * ranks - 1.0
        factor = 1.0 + gamma * z
    else:
        raise ValueError(mode)

    # Apply
    factor_b = factor[:, np.newaxis, :]  # (n_files, 1, 234)
    return preds_3d * factor_b


def main():
    print("Loading OOF…")
    d = np.load("kaggle_model/sed_oof_preds.npz", allow_pickle=True)
    preds = d["predictions"]; labels = d["labels"]; row_ids = d["row_ids"]

    files_to_idx = {}
    for i, rid in enumerate(row_ids):
        parts = rid.rsplit("_", 1)
        fn, end_sec = parts[0], int(parts[1])
        files_to_idx.setdefault(fn, []).append((i, end_sec))
    for fn in files_to_idx:
        files_to_idx[fn].sort(key=lambda x: x[1])
    files_to_idx = {fn: v for fn, v in files_to_idx.items() if len(v) == 12}
    n_files = len(files_to_idx)
    print(f"  {n_files} soundscapes × 12 chunks")

    file_order = sorted(files_to_idx.keys())
    pred_arr = np.zeros((n_files * 12, 234), dtype=np.float32)
    label_arr = np.zeros((n_files * 12, 234), dtype=np.uint8)
    for fi, fn in enumerate(file_order):
        for ci, (orig_idx, _) in enumerate(files_to_idx[fn]):
            pred_arr[fi*12 + ci] = preds[orig_idx]
            label_arr[fi*12 + ci] = labels[orig_idx]

    preds_3d = pred_arr.reshape(n_files, 12, 234)

    base_auc, n_classes = macro_auc(label_arr, pred_arr)
    print(f"\nBaseline macro AUC: {base_auc:.4f} over {n_classes} classes\n")

    # Sweep
    print("=== zscore mode ===")
    print(f"{'topK':>5} {'γ':>5} {'AUC':>8} {'Δ':>8}")
    for K_top in [1, 2, 3, 4]:
        for gamma in [0.10, 0.20, 0.30, 0.50, 1.00]:
            filtered = soundscape_bayes_filter(preds_3d, K_top, gamma, mode="zscore")
            auc, _ = macro_auc(label_arr, filtered.reshape(-1, 234))
            d = auc - base_auc
            marker = " *" if d > 0.0001 else ("  -" if d < -0.0005 else "")
            print(f"{K_top:>5} {gamma:>5.2f} {auc:>8.4f} {d:>+8.4f}{marker}")

    print("\n=== rank mode ===")
    print(f"{'topK':>5} {'γ':>5} {'AUC':>8} {'Δ':>8}")
    for K_top in [1, 2, 3, 4]:
        for gamma in [0.10, 0.20, 0.30, 0.50, 1.00]:
            filtered = soundscape_bayes_filter(preds_3d, K_top, gamma, mode="rank")
            auc, _ = macro_auc(label_arr, filtered.reshape(-1, 234))
            d = auc - base_auc
            marker = " *" if d > 0.0001 else ("  -" if d < -0.0005 else "")
            print(f"{K_top:>5} {gamma:>5.2f} {auc:>8.4f} {d:>+8.4f}{marker}")


if __name__ == "__main__":
    main()
