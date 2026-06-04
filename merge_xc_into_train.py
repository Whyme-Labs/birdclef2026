"""V181 Phase 3: Merge Pantanal XC audio into train.csv format with source='XC'.

Output: data/train_xc_merged.csv with rows for:
  - All existing train_merged.csv rows (source 2026 + 2025)
  - New XC rows (source 'XC') with primary_label mapped from scientific_name
"""
import pandas as pd
import json
from pathlib import Path


def main():
    base = Path("data/train_merged.csv")
    if not base.exists():
        base = Path("data/train.csv")
        print(f"Using base: {base} (no train_merged.csv)")
    df_base = pd.read_csv(base)
    print(f"Base rows: {len(df_base)}")

    if "source" not in df_base.columns:
        df_base["source"] = "2026"

    # Load XC manifest
    xc_manifest = Path("data_external/xc_pantanal_audio/manifest.csv")
    if not xc_manifest.exists():
        print(f"ERROR: {xc_manifest} not found")
        return
    xc = pd.read_csv(xc_manifest)
    print(f"XC manifest rows: {len(xc)}")

    # Load taxonomy
    tax = pd.read_csv("data/taxonomy.csv")
    sci_to_tax = dict(zip(
        tax["scientific_name"].str.lower(),
        tax[["primary_label", "scientific_name", "common_name", "class_name", "inat_taxon_id"]].to_dict("records")
    ))

    rows = []
    audio_root = Path("data_external/xc_pantanal_audio")
    for _, r in xc.iterrows():
        sci = (str(r.get("gen", "")) + " " + str(r.get("sp", ""))).strip()
        sci_lc = sci.lower()
        if sci_lc not in sci_to_tax:
            continue

        tax_info = sci_to_tax[sci_lc]
        label = str(tax_info["primary_label"])
        rec_id = r["id"]
        # Find actual file (could be mp3 or ogg)
        cand_paths = list((audio_root / label).glob(f"XC{rec_id}.*"))
        if not cand_paths:
            continue
        rel_path = f"{label}/{cand_paths[0].name}"

        rows.append({
            "primary_label": label,
            "secondary_labels": "[]",
            "type": str(r.get("type", "")) or "[]",
            "latitude": r.get("lat", ""),
            "longitude": r.get("lon", ""),
            "scientific_name": tax_info["scientific_name"],
            "common_name": tax_info["common_name"],
            "class_name": tax_info["class_name"],
            "inat_taxon_id": tax_info["inat_taxon_id"],
            "author": r.get("rec", ""),
            "license": r.get("lic", ""),
            "rating": 0.0,  # XC doesn't expose rating directly
            "url": r.get("url", ""),
            "filename": rel_path,
            "collection": "XenoCanto",
            "source": "XC",
        })

    print(f"XC rows added: {len(rows)}")
    df_xc = pd.DataFrame(rows)

    # Concat
    df_out = pd.concat([df_base, df_xc], ignore_index=True)
    print(f"\nMerged total: {len(df_out)}")
    print(df_out["source"].value_counts())

    out = Path("data/train_xc_merged.csv")
    df_out.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Per-species coverage
    print("\nPer-species recording count (with vs without XC):")
    base_counts = df_base.groupby("primary_label").size()
    out_counts = df_out.groupby("primary_label").size()
    diff = (out_counts - base_counts).fillna(out_counts).astype(int)
    boosted = diff[diff > 0]
    print(f"Species boosted by XC: {len(boosted)}")
    print(f"  Top 10 boosts: {boosted.sort_values(ascending=False).head(10).to_string()}")
    print(f"  Mean boost (boosted species): +{boosted.mean():.1f} recs")


if __name__ == "__main__":
    main()
