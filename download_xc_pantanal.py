"""V181 Phase 2: Download Pantanal XC recordings matching 2026 taxonomy.

Filters to:
  - Species in our 234 2026 taxonomy
  - Quality grade A or B (skip C/D/E)
  - Length under 5 minutes (avoid huge files)

Saves to data_external/xc_pantanal_audio/<primary_label>/<id>.ogg
Mirroring train_audio structure for direct training-pipeline integration.
"""
import argparse, time, os, json
from pathlib import Path
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_length(s):
    """e.g. '0:47' -> 47 seconds, '2:30' -> 150 seconds."""
    try:
        if not s or pd.isna(s): return 0
        parts = str(s).split(":")
        if len(parts) == 2: return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return int(parts[0])
    except: return 0


def download_one(rec_id, url, out_path, api_key, retries=2):
    if out_path.exists() and out_path.stat().st_size > 1024:
        return "exists", 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "BirdCLEF2026Research/1.0"}
    full_url = f"{url}?key={api_key}" if "?" not in url else f"{url}&key={api_key}"
    for attempt in range(retries):
        try:
            r = requests.get(full_url, headers=headers, timeout=60, stream=True)
            r.raise_for_status()
            size = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=64*1024):
                    if chunk:
                        f.write(chunk)
                        size += len(chunk)
            return "ok", size
        except Exception as e:
            if attempt == retries - 1:
                return f"err:{e}", 0
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_external/xc_pantanal_manifest.csv")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--out_dir", default="data_external/xc_pantanal_audio")
    ap.add_argument("--api_key", default=os.environ.get("XC_API_KEY", ""))
    ap.add_argument("--quality", default="A,B", help="comma-separated quality grades")
    ap.add_argument("--max_length_sec", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: XC_API_KEY required")
        return

    df = pd.read_csv(args.manifest)
    df["scientific_name"] = (df["gen"].astype(str) + " " + df["sp"].astype(str)).str.strip()
    print(f"Manifest: {len(df)} records")

    tax = pd.read_csv(args.taxonomy)
    tax["scientific_name_lc"] = tax["scientific_name"].str.lower()
    sci_to_label = dict(zip(tax["scientific_name_lc"], tax["primary_label"].astype(str)))

    df["sci_lc"] = df["scientific_name"].str.lower()
    df["primary_label"] = df["sci_lc"].map(sci_to_label)
    df["length_sec"] = df["length"].apply(parse_length)

    # Filter
    matched = df[df["primary_label"].notna()].copy()
    print(f"  Matched 2026 taxonomy: {len(matched)}")

    quality_set = set(q.strip().upper() for q in args.quality.split(","))
    quality_filtered = matched[matched["q"].astype(str).str.upper().isin(quality_set)]
    print(f"  Quality {sorted(quality_set)}: {len(quality_filtered)}")

    length_filtered = quality_filtered[
        (quality_filtered["length_sec"] > 0) &
        (quality_filtered["length_sec"] <= args.max_length_sec)
    ]
    print(f"  Length ≤{args.max_length_sec}s: {len(length_filtered)}")

    final = length_filtered
    if args.limit > 0:
        final = final.head(args.limit)
    print(f"\nDownloading {len(final)} recordings...")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save filtered manifest
    manifest_out = out_dir / "manifest.csv"
    final.to_csv(manifest_out, index=False)
    print(f"Saved filtered manifest: {manifest_out}")

    # Build download tasks
    tasks = []
    for _, row in final.iterrows():
        rec_id = row["id"]
        url = row["file"]
        label = row["primary_label"]
        # Try to use file-name extension, default ogg
        fname = row.get("file-name", "")
        ext = "ogg"
        if fname and "." in str(fname):
            cand = str(fname).rsplit(".", 1)[1].lower()
            if cand in ("ogg", "wav", "mp3", "flac"):
                ext = cand
        out_path = out_dir / str(label) / f"XC{rec_id}.{ext}"
        tasks.append((rec_id, url, out_path))

    # Download in parallel
    t0 = time.time()
    n_ok = n_err = n_skip = 0
    total_size = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, rid, url, op, args.api_key): rid for rid, url, op in tasks}
        for fi, fut in enumerate(as_completed(futures)):
            status, size = fut.result()
            if status == "ok":
                n_ok += 1
                total_size += size
            elif status == "exists":
                n_skip += 1
            else:
                n_err += 1
            if (fi + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (fi + 1) / elapsed
                print(f"  {fi+1}/{len(tasks)} [ok={n_ok} skip={n_skip} err={n_err}] "
                      f"{total_size/1e6:.0f}MB rate={rate:.1f}/s")

    print(f"\nTotal: {time.time()-t0:.0f}s, ok={n_ok} skip={n_skip} err={n_err}")
    print(f"Total downloaded: {total_size/1e6:.0f}MB")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
