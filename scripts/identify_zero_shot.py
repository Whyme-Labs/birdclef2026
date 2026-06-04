"""Identify species in taxonomy.csv that have ZERO entries in train.csv."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/home/soh/birdclef-2026/data")
TAXONOMY_CSV = DATA_DIR / "taxonomy.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
OUT_CSV = DATA_DIR / "zero_shot_species.csv"


def main() -> None:
    tax = pd.read_csv(TAXONOMY_CSV)
    train = pd.read_csv(TRAIN_CSV, low_memory=False)

    tax["primary_label"] = tax["primary_label"].astype(str)
    train["primary_label"] = train["primary_label"].astype(str)

    counts = train.groupby("primary_label").size().rename("n_train").reset_index()
    merged = tax.merge(counts, on="primary_label", how="left")
    merged["n_train"] = merged["n_train"].fillna(0).astype(int)

    print(f"Total taxonomy species : {len(tax)}")
    print(f"Total train recordings : {len(train)}")
    print(f"Species with >=1 rec   : {(merged['n_train'] > 0).sum()}")
    print(f"Zero-shot species      : {(merged['n_train'] == 0).sum()}")

    zero_shot = merged.loc[merged["n_train"] == 0, ["primary_label", "scientific_name", "class_name"]]
    zero_shot = zero_shot.sort_values(["class_name", "scientific_name"]).reset_index(drop=True)
    zero_shot.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(zero_shot)} rows -> {OUT_CSV}")

    by_class = zero_shot["class_name"].value_counts()
    print("\nZero-shot by class:")
    for cls, n in by_class.items():
        print(f"  {cls:12s}: {n}")


if __name__ == "__main__":
    main()
