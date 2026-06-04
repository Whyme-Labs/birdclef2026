"""V161-v2: graph propagation using V137 PREDICTION similarity (not AudioMAE)."""
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


def graph_propagate_pred(preds_3d, tau, alpha, K):
    """Propagate using PREDICTION similarity as edge weights."""
    out = preds_3d.copy()
    for fi in range(preds_3d.shape[0]):
        p0 = preds_3d[fi].astype(np.float32)  # (12, 234)
        # Normalize predictions to unit norm for cosine similarity
        p_norm = p0 / np.maximum(np.linalg.norm(p0, axis=1, keepdims=True), 1e-9)
        sim = p_norm @ p_norm.T  # (12, 12)
        logits = sim * tau
        logits = logits - logits.max(axis=1, keepdims=True)
        W = np.exp(logits)
        W = W / W.sum(axis=1, keepdims=True)
        p = p0.copy()
        for _ in range(K):
            p = (1 - alpha) * p0 + alpha * (W @ p)
        out[fi] = p
    return out


def graph_propagate_topk_overlap(preds_3d, K_top, alpha, K_iter):
    """Propagate using TOP-K species overlap as edge weights."""
    out = preds_3d.copy()
    for fi in range(preds_3d.shape[0]):
        p0 = preds_3d[fi].astype(np.float32)  # (12, 234)
        # Top-K species per chunk
        topk_idx = np.argsort(-p0, axis=1)[:, :K_top]  # (12, K)
        # Build sets
        sets = [set(topk_idx[i].tolist()) for i in range(12)]
        W = np.zeros((12, 12), dtype=np.float32)
        for i in range(12):
            for j in range(12):
                inter = len(sets[i] & sets[j])
                W[i, j] = inter / K_top
        # Row-normalize
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
        p = p0.copy()
        for _ in range(K_iter):
            p = (1 - alpha) * p0 + alpha * (W @ p)
        out[fi] = p
    return out


def main():
    print("Loading OOF…")
    d = np.load("kaggle_model/sed_oof_preds.npz", allow_pickle=True)
    preds = d["predictions"]
    labels = d["labels"]
    row_ids = d["row_ids"]

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

    # Method 1: prediction-cos similarity propagation
    print("=== Method 1: prediction cosine similarity ===")
    print(f"{'τ':>5} {'α':>5} {'K':>3} {'AUC':>8} {'Δ':>8}")
    for tau in [4.0, 8.0, 16.0, 32.0, 64.0]:
        for alpha in [0.10, 0.20, 0.30, 0.50]:
            for K in [1, 2, 3]:
                propagated = graph_propagate_pred(preds_3d, tau, alpha, K)
                auc, _ = macro_auc(label_arr, propagated.reshape(-1, 234))
                d = auc - base_auc
                marker = " *" if d > 0.0001 else ("  -" if d < -0.0005 else "")
                print(f"{tau:>5.1f} {alpha:>5.2f} {K:>3} {auc:>8.4f} {d:>+8.4f}{marker}")

    # Method 2: top-K overlap propagation
    print("\n=== Method 2: top-K species overlap ===")
    print(f"{'topK':>5} {'α':>5} {'K':>3} {'AUC':>8} {'Δ':>8}")
    for K_top in [3, 5, 10, 20]:
        for alpha in [0.10, 0.20, 0.30, 0.50]:
            for K_iter in [1, 2, 3]:
                propagated = graph_propagate_topk_overlap(preds_3d, K_top, alpha, K_iter)
                auc, _ = macro_auc(label_arr, propagated.reshape(-1, 234))
                d = auc - base_auc
                marker = " *" if d > 0.0001 else ("  -" if d < -0.0005 else "")
                print(f"{K_top:>5} {alpha:>5.2f} {K_iter:>3} {auc:>8.4f} {d:>+8.4f}{marker}")


if __name__ == "__main__":
    main()
