"""
Merge BirdCLEF 2026 training data with external BirdCLEF 2025 data.

Strategy:
  - Load existing 2026 train.csv (35549 rows, 206 species present).
  - Load external_data_overlap.csv, filter to available=True AND source_year=2025.
  - Map 2025 files to our canonical primary_label via the target_pl column.
  - Produce train_merged.csv with a "source" column indicating 2026 or 2025,
    so dataset.py can resolve to the correct audio_dir.

The 2026 training script passes `data_dir / "train_audio"` as audio_dir, and
BirdCLEFDataset appends `row["filename"]` to it. For 2025 rows we store a
relative path under data_external/ and a source marker so the dataset can
pick the correct base.

Output columns mirror train.csv plus a `source` column:
  primary_label, secondary_labels, type, latitude, longitude, scientific_name,
  common_name, class_name, inat_taxon_id, author, license, rating, url,
  filename, collection, source

For 2025 rows:
  - primary_label = target_pl (our canonical label)
  - filename       = <species>/<audio_file>.ogg (relative to data_external/2025_audio/birdclef-2025/train_audio/)
  - source         = "2025"
  - other columns filled with sensible defaults / pulled from overlap CSV
"""
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_2026 = ROOT / "data"
DATA_EXT = ROOT / "data_external"
TRAIN_CSV = DATA_2026 / "train.csv"
OVERLAP_CSV = DATA_EXT / "external_data_overlap.csv"
OUT_CSV = DATA_2026 / "train_merged.csv"

# The audio under data_external/2025_audio/birdclef-2025/train_audio/<species>/<file>.ogg
EXT_2025_BASE = "2025_audio/birdclef-2025/train_audio"


def main():
    print(f"Reading {TRAIN_CSV}")
    df_2026 = pd.read_csv(TRAIN_CSV)
    df_2026["source"] = "2026"
    print(f"  2026 rows: {len(df_2026)} | species: {df_2026.primary_label.nunique()}")

    print(f"Reading {OVERLAP_CSV}")
    df_overlap = pd.read_csv(OVERLAP_CSV)
    avail_2025 = df_overlap[
        (df_overlap.available == True) & (df_overlap.source_year == 2025)
    ].copy()
    print(f"  2025 available rows: {len(avail_2025)} | target species: {avail_2025.target_pl.nunique()}")

    # Build a 2025 DataFrame with the same schema as train.csv plus source.
    # source_path is like "2025_audio/birdclef-2025/train_audio/22973/XC167037.ogg".
    # Strip the EXT_2025_BASE prefix so filename becomes "22973/XC167037.ogg",
    # consistent with 2026 train.csv format ("<label>/<file>.ogg").
    prefix = EXT_2025_BASE + "/"
    def rel_filename(p):
        p = str(p)
        if p.startswith(prefix):
            return p[len(prefix):]
        # Fallback: return as-is (should not happen).
        return p

    df_2025 = pd.DataFrame({
        "primary_label": avail_2025["target_pl"].astype(str),
        "secondary_labels": "[]",
        "type": "[]",
        "latitude": pd.NA,
        "longitude": pd.NA,
        "scientific_name": avail_2025["scientific_name"].fillna(""),
        "common_name": "",
        "class_name": "",
        "inat_taxon_id": pd.NA,
        "author": "",
        "license": "",
        "rating": 0.0,
        "url": "",
        "filename": avail_2025["source_path"].map(rel_filename),
        "collection": "birdclef-2025",
        "source": "2025",
    })

    # Verify columns match.
    expected_cols = df_2026.columns.tolist()
    missing = set(expected_cols) - set(df_2025.columns)
    extra = set(df_2025.columns) - set(expected_cols)
    if missing or extra:
        print(f"Column mismatch — missing {missing} extra {extra}")
    df_2025 = df_2025[expected_cols]

    # Validate a few file paths actually exist on disk before writing.
    check_base = DATA_EXT / EXT_2025_BASE
    check_df = df_2025.sample(min(50, len(df_2025)), random_state=0)
    missing_files = 0
    for f in check_df["filename"]:
        p = check_base / f
        if not p.exists():
            missing_files += 1
            if missing_files <= 3:
                print(f"  MISSING: {p}")
    print(f"  File-existence check: {missing_files}/{len(check_df)} missing")

    # Concatenate.
    df_merged = pd.concat([df_2026, df_2025], ignore_index=True)
    print(f"Merged rows: {len(df_merged)} | species: {df_merged.primary_label.nunique()}")
    print(f"Source counts:")
    print(df_merged.groupby("source").size())

    df_merged.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
