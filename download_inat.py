#!/usr/bin/env python
"""Download audio recordings from iNaturalist for 3 amphibian species."""
import json
import os
import sys
import time
from pathlib import Path

import requests

SPECIES = [
    ("1491113", "Adenomera guarani"),
    ("25073", "Chiasmocleis mehelyi"),
    ("517063", "Pithecopus azureus"),
]

BASE_DIR = Path("/home/soh/birdclef-2026/data_external/inaturalist")
MAX_PER_SPECIES = 50
MAX_BYTES = 30 * 1024 * 1024  # 30MB
RATE = 1.05  # seconds between requests
HEADERS = {
    "User-Agent": "BirdCLEF2026-research/1.0 (wmhy.tech@gmail.com)"
}


def rate_sleep():
    time.sleep(RATE)


def get_taxon_id(scientific_name):
    """Look up taxon_id by scientific name."""
    url = "https://api.inaturalist.org/v1/taxa"
    params = {"q": scientific_name, "rank": "species"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        rate_sleep()
        if r.status_code != 200:
            print(f"    taxa lookup status={r.status_code}")
            return None
        data = r.json()
        results = data.get("results", [])
        # Prefer exact match
        for row in results:
            if row.get("name", "").lower() == scientific_name.lower():
                return row.get("id"), row.get("observations_count", 0)
        if results:
            return results[0].get("id"), results[0].get("observations_count", 0)
    except Exception as e:
        print(f"    taxa lookup error: {e}")
    return None


def fetch_observations(taxon_id, max_needed):
    """Fetch observations with sounds for this taxon."""
    observations = []
    page = 1
    per_page = 200
    while len(observations) < max_needed * 3:  # over-fetch since some may fail
        url = "https://api.inaturalist.org/v1/observations"
        params = {
            "taxon_id": taxon_id,
            "has_sound": "true",
            "per_page": per_page,
            "page": page,
            "order": "desc",
            "order_by": "created_at",
        }
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=45)
            rate_sleep()
            if r.status_code != 200:
                print(f"    obs page {page} status={r.status_code}")
                break
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            observations.extend(results)
            total = data.get("total_results", 0)
            print(f"    page {page}: got {len(results)} (total={total})")
            if len(observations) >= total:
                break
            if page * per_page >= total:
                break
            page += 1
            if page > 5:  # safety cap
                break
        except Exception as e:
            print(f"    obs fetch error: {e}")
            break
    return observations


def pick_best_url(sound):
    """Extract best available file URL from a sound entry."""
    if not isinstance(sound, dict):
        return None
    # Primary field
    fu = sound.get("file_url")
    if fu:
        return fu
    # Some responses nest
    su = sound.get("sound", {})
    if isinstance(su, dict):
        fu = su.get("file_url")
        if fu:
            return fu
    return None


def pick_extension(url, content_type):
    """Figure out file extension."""
    # Extension from URL
    low = url.split("?")[0].lower()
    for ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm"):
        if low.endswith(ext):
            return ext
    # From content type
    if content_type:
        ct = content_type.lower()
        if "mpeg" in ct or "mp3" in ct:
            return ".mp3"
        if "wav" in ct:
            return ".wav"
        if "mp4" in ct or "m4a" in ct:
            return ".m4a"
        if "ogg" in ct:
            return ".ogg"
        if "flac" in ct:
            return ".flac"
    return ".mp3"


def download_file(url, dest_prefix):
    """Stream-download, cap at MAX_BYTES."""
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
            if r.status_code != 200:
                return None, f"status={r.status_code}"
            ct = r.headers.get("Content-Type", "")
            ext = pick_extension(url, ct)
            dest = dest_prefix.with_suffix(ext)
            cl = r.headers.get("Content-Length")
            if cl and int(cl) > MAX_BYTES:
                return None, f"too_big={cl}"
            total = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_BYTES:
                        f.close()
                        dest.unlink(missing_ok=True)
                        return None, "exceeded_cap"
                    f.write(chunk)
            if total < 2048:
                dest.unlink(missing_ok=True)
                return None, "too_small"
            return (dest, total), None
    except Exception as e:
        return None, f"err={e}"


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    total_bytes = 0

    for primary_label, sci_name in SPECIES:
        print(f"\n=== {sci_name} (primary_label={primary_label}) ===")
        out_dir = BASE_DIR / primary_label
        out_dir.mkdir(parents=True, exist_ok=True)

        taxon_res = get_taxon_id(sci_name)
        if not taxon_res:
            print(f"  NO taxon found for {sci_name}")
            summary.append({
                "primary_label": primary_label,
                "scientific_name": sci_name,
                "taxon_id": None,
                "obs_total": 0,
                "downloaded": 0,
                "bytes": 0,
                "note": "no taxon match",
            })
            continue
        taxon_id, obs_count = taxon_res
        print(f"  taxon_id={taxon_id}  obs_count={obs_count}")

        observations = fetch_observations(taxon_id, MAX_PER_SPECIES)
        print(f"  total observations fetched: {len(observations)}")

        downloaded = 0
        sp_bytes = 0
        errors = {}
        for obs in observations:
            if downloaded >= MAX_PER_SPECIES:
                break
            obs_id = obs.get("id")
            sounds = obs.get("sounds") or []
            if not sounds:
                continue
            for idx, sound in enumerate(sounds):
                if downloaded >= MAX_PER_SPECIES:
                    break
                url = pick_best_url(sound)
                if not url:
                    continue
                # Convert http -> https if relative protocol
                if url.startswith("//"):
                    url = "https:" + url
                prefix = out_dir / f"obs{obs_id}_s{idx}"
                # Skip if already exists
                existing = list(out_dir.glob(f"obs{obs_id}_s{idx}.*"))
                if existing:
                    continue
                res, err = download_file(url, prefix)
                rate_sleep()
                if err:
                    errors[err] = errors.get(err, 0) + 1
                    continue
                dest, size = res
                downloaded += 1
                sp_bytes += size
                if downloaded % 5 == 0 or downloaded <= 3:
                    print(f"    [{downloaded}/{MAX_PER_SPECIES}] {dest.name} ({size/1024:.1f} KB)")
        print(f"  downloaded {downloaded} files, {sp_bytes/1024/1024:.2f} MB")
        if errors:
            print(f"  errors: {errors}")
        summary.append({
            "primary_label": primary_label,
            "scientific_name": sci_name,
            "taxon_id": taxon_id,
            "obs_total": len(observations),
            "downloaded": downloaded,
            "bytes": sp_bytes,
            "errors": errors,
        })
        total_bytes += sp_bytes

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"total_bytes={total_bytes} ({total_bytes/1024/1024:.2f} MB)")

    # Write summary json for post-processing
    with open(BASE_DIR / "_summary.json", "w") as f:
        json.dump({"species": summary, "total_bytes": total_bytes}, f, indent=2)


if __name__ == "__main__":
    main()
