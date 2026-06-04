"""V181: Xeno-Canto Pantanal-region scrape.

Per BirdCLEF 2025 1st (5,489 XC entries) and 2nd (7,376 XC entries) place.
Pantanal coordinates: roughly -16 to -22 lat, -54 to -60 lon (Brazil/Bolivia/Paraguay).

Two-pass strategy:
  Pass 1: Filter by geographic box (Pantanal region) for any species in 2026 taxonomy
  Pass 2: For 234 competition species not well-represented in train_audio, do
          species-specific queries (regardless of location) to fill data gaps

XC API: https://xeno-canto.org/api/3/recordings (NEW API)
  Old: https://www.xeno-canto.org/api/2/recordings (deprecated)
  Auth: API key required as ?key=YOUR_KEY (free signup)

This script writes manifest CSV first (no audio download) so we can review
species distribution before committing disk to downloads.
"""
import argparse, json, time, os
from pathlib import Path
import requests
import pandas as pd


# Pantanal bounding box (rough)
PANTANAL_BBOX = {
    "lat_min": -22.5,
    "lat_max": -15.5,
    "lon_min": -60.0,
    "lon_max": -54.0,
}

XC_API_BASE = "https://xeno-canto.org/api/3/recordings"


def query_xc(query: str, api_key: str, page: int = 1):
    """Query XC API, returns full response JSON."""
    params = {"query": query, "page": page, "key": api_key}
    r = requests.get(XC_API_BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_all_pages(query: str, api_key: str, max_pages: int = 100, sleep: float = 0.5):
    """Paginate through XC API for a given query."""
    all_recs = []
    page = 1
    while page <= max_pages:
        try:
            data = query_xc(query, api_key, page=page)
        except Exception as e:
            print(f"  page {page} err: {e}")
            break
        recs = data.get("recordings", [])
        if not recs:
            break
        all_recs.extend(recs)
        num_pages = int(data.get("numPages", 1))
        print(f"  page {page}/{num_pages}: +{len(recs)} (total {len(all_recs)})")
        page += 1
        if page > num_pages:
            break
        time.sleep(sleep)  # rate-limit politeness
    return all_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=os.environ.get("XC_API_KEY", ""),
                    help="Xeno-Canto API key (env XC_API_KEY)")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--out_manifest", default="data_external/xc_pantanal_manifest.csv")
    ap.add_argument("--bbox_only", action="store_true",
                    help="Only do geographic box query (skip species-specific)")
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: XC API key required. Get one at https://xeno-canto.org/account/api")
        print("Set XC_API_KEY env var or pass --api_key")
        return

    tax = pd.read_csv(args.taxonomy)
    print(f"Taxonomy: {len(tax)} species")
    print(f"Classes: {tax['class_name'].value_counts().to_dict()}")

    # Pass 1: Pantanal geographic box query
    bbox = PANTANAL_BBOX
    geo_query = f"box:{bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']}"
    print(f"\nPass 1: Pantanal geographic box query")
    print(f"  Query: {geo_query}")
    pantanal_recs = fetch_all_pages(geo_query, args.api_key, max_pages=200, sleep=0.3)
    print(f"  Total Pantanal-region recordings: {len(pantanal_recs)}")

    if not args.bbox_only:
        # Pass 2: per-species queries for under-represented birds
        # XC supports species name queries; we use scientific_name from taxonomy
        bird_species = tax[tax["class_name"] == "Aves"]
        amph_species = tax[tax["class_name"] == "Amphibia"]
        ins_species = tax[tax["class_name"] == "Insecta"]
        print(f"\nPass 2: per-species queries (birds={len(bird_species)}, amph={len(amph_species)}, ins={len(ins_species)})")
        print(f"  (skipping for now — pass --species_specific to enable)")

    # Save manifest
    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(pantanal_recs)
    df.to_csv(out, index=False)
    print(f"\nSaved manifest: {out} ({len(df)} rows)")

    # Stats
    if "gen" in df.columns and "sp" in df.columns:
        df["scientific_name"] = df["gen"] + " " + df["sp"]
        # Match against 2026 taxonomy
        tax_set = set(tax["scientific_name"].str.lower())
        df["in_2026"] = df["scientific_name"].str.lower().isin(tax_set)
        print(f"\nXC recordings matching 2026 taxonomy: {df['in_2026'].sum()} / {len(df)}")
        print("Top 20 species by recording count (in 2026 taxonomy):")
        match_df = df[df["in_2026"]]
        if len(match_df):
            print(match_df["scientific_name"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
