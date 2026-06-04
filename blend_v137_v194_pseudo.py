"""Combine V137 ensemble pseudo + V194_iter1 pseudo into a soup for iter 2 training.

Multi-iter NS recipe step 2.5: blend the two pseudo sources element-wise.
weight w * V137 + (1-w) * V194_iter1, w=0.5 (equal contribution).

Output: pseudo_labels_v194_iter2/raw_predictions.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v137", default="pseudo_labels_v137/raw_predictions.csv")
    ap.add_argument("--v194", default="pseudo_labels_v194_iter1/raw_predictions.csv")
    ap.add_argument("--out", default="pseudo_labels_v194_iter2_blend/raw_predictions.csv")
    ap.add_argument("--w_v137", type=float, default=0.5)
    args = ap.parse_args()

    print(f"Loading V137 pseudo: {args.v137}")
    df137 = pd.read_csv(args.v137)
    print(f"  {len(df137)} rows × {len(df137.columns)} cols")

    print(f"Loading V194_iter1 pseudo: {args.v194}")
    df194 = pd.read_csv(args.v194)
    print(f"  {len(df194)} rows × {len(df194.columns)} cols")

    # Align by (file, end_time)
    label_cols = [c for c in df137.columns if c not in ("file", "end_time")]
    df137 = df137.set_index(["file", "end_time"]).sort_index()
    df194 = df194.set_index(["file", "end_time"]).sort_index()

    common = df137.index.intersection(df194.index)
    print(f"  common chunks: {len(common)}")

    a = df137.loc[common, label_cols].values.astype(np.float32)
    b = df194.loc[common, label_cols].values.astype(np.float32)

    w = args.w_v137
    blend = w * a + (1 - w) * b
    print(f"Blend = {w:.2f}*V137 + {1-w:.2f}*V194_iter1")
    print(f"  V137 mean={a.mean():.4f}, V194 mean={b.mean():.4f}, blend mean={blend.mean():.4f}")

    out = pd.DataFrame(blend, columns=label_cols, index=common).reset_index()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
