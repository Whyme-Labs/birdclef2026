"""V166: Build species co-occurrence lift matrix from training soundscape labels.

Lift(A, B) = P(A, B) / (P(A) × P(B))
  • > 1: positively correlated (often co-occur)
  • = 1: independent
  • < 1: negatively correlated

At test time, V166 will use this matrix to boost species predictions where
correlated species are also predicted high — empirical Bayesian message passing
in species space (analogous to belief propagation on a sparse Bayesian network).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundscape_csv", default="data/train_soundscapes_labels.csv")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--out", default="kaggle_model/cooccurrence_v166.npz")
    ap.add_argument("--smoothing", type=float, default=1.0,
                    help="Laplace smoothing for joint counts")
    args = ap.parse_args()

    tax = pd.read_csv(args.taxonomy)
    label_cols = sorted(tax["primary_label"].astype(str).tolist())
    l2i = {c: i for i, c in enumerate(label_cols)}
    C = len(label_cols)
    print(f"Classes: {C}")

    sc = pd.read_csv(args.soundscape_csv)
    def parse_lbl(x):
        if pd.isna(x) or x == "nan": return set()
        return set(t.strip() for t in str(x).split(";") if t.strip())
    sc["lbl"] = sc["primary_label"].astype(str).apply(parse_lbl)
    print(f"rows: {len(sc)}")

    # Per-chunk multi-hot Y
    Y = np.zeros((len(sc), C), dtype=np.float32)
    for i, labels in enumerate(sc["lbl"]):
        for lbl in labels:
            if lbl in l2i:
                Y[i, l2i[lbl]] = 1.0
    n_per_class = Y.sum(axis=0)
    n_chunks = len(Y)
    print(f"Per-class positive counts: min={n_per_class.min()} median={int(np.median(n_per_class))} max={n_per_class.max()}")
    print(f"Classes with ≥1 positive: {(n_per_class > 0).sum()}")

    # Marginal probabilities (with smoothing)
    P_marginal = (n_per_class + args.smoothing) / (n_chunks + 2 * args.smoothing)

    # Joint probabilities (co-occurrence counts)
    # Joint[a, b] = sum over chunks of Y[chunk, a] * Y[chunk, b]
    P_joint = (Y.T @ Y + args.smoothing) / (n_chunks + 2 * args.smoothing)  # (C, C)
    # P_joint[a, a] is just P_marginal[a] (after smoothing)

    # Lift matrix: P(a, b) / (P(a) × P(b))
    P_outer = P_marginal[:, None] * P_marginal[None, :]
    lift = P_joint / np.maximum(P_outer, 1e-12)  # (C, C)
    np.fill_diagonal(lift, 1.0)  # self-lift = 1 (no self-boost)

    # For species with no positives, lift to themselves and others = 1 (neutral)
    no_pos = n_per_class == 0
    lift[no_pos, :] = 1.0
    lift[:, no_pos] = 1.0
    np.fill_diagonal(lift, 1.0)

    # Mask: species with enough data for reliable lift (≥10 positives)
    reliable = (n_per_class >= 10).astype(bool)
    print(f"Reliable species (≥10 positives): {reliable.sum()}/{C}")

    # Set lift for unreliable species to/from = 1 (neutral)
    lift[~reliable, :] = 1.0
    lift[:, ~reliable] = 1.0
    np.fill_diagonal(lift, 1.0)

    # Clip lift to robust range — avoid overconfidence on noisy estimates
    # Pairs with very high lift but small joint counts are artifacts
    joint_count = (Y.T @ Y).astype(int)
    weak_evidence = joint_count < 5
    lift[weak_evidence] = np.clip(lift[weak_evidence], 0.5, 2.0)
    lift = np.clip(lift, 0.1, 5.0)
    np.fill_diagonal(lift, 1.0)

    # Stats
    off_diag = lift[~np.eye(C, dtype=bool)]
    print(f"\nLift stats (off-diagonal, all species):")
    print(f"  mean={off_diag.mean():.3f}, std={off_diag.std():.3f}")
    print(f"  >2.0 (strong positive): {(off_diag > 2.0).sum()}")
    print(f"  >5.0 (very strong positive): {(off_diag > 5.0).sum()}")
    print(f"  <0.5 (negative correlation): {(off_diag < 0.5).sum()}")

    # Top correlated pairs
    pairs = []
    for i in range(C):
        for j in range(i+1, C):
            if reliable[i] and reliable[j] and lift[i, j] > 2.0:
                pairs.append((label_cols[i], label_cols[j], lift[i, j], int(P_joint[i, j] * n_chunks)))
    pairs.sort(key=lambda x: -x[2])
    print(f"\nTop 10 high-lift pairs:")
    for p in pairs[:10]:
        print(f"  {p[0]} ↔ {p[1]}: lift={p[2]:.2f} ({p[3]} co-occurrences)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        lift=lift.astype(np.float32),
                        marginal=P_marginal.astype(np.float32),
                        reliable=reliable,
                        n_per_class=n_per_class.astype(np.int32),
                        label_cols=np.array(label_cols))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
