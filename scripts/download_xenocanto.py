"""Download XenoCanto recordings for zero-shot BIRD species.

XenoCanto API v2 was retired. The v3 endpoint now requires an authenticated
`key` parameter. Provide it via the ``XC_API_KEY`` environment variable or the
``--api-key`` flag. Without a key the script explains how to get one and exits.

Usage:
    XC_API_KEY=<key> python scripts/download_xenocanto.py \
        --zero-shot /home/soh/birdclef-2026/data/zero_shot_species.csv \
        --out-dir   /home/soh/birdclef-2026/data_external/xenocanto \
        --max-per-species 20

Licensing: we persist the XenoCanto `lic`, `rec` (recorder), and URL into a
per-species ``metadata.csv`` so every recording has proper attribution.
Recordings whose license cannot be resolved are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests

API_V3 = "https://xeno-canto.org/api/3/recordings"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "BirdCLEF-Research/1.0 (contact: wmhy.tech@gmail.com)"
)
MAX_BYTES = 20 * 1024 * 1024  # 20 MB per recording
DEFAULT_MAX_SECONDS = 60      # prefer <=60s clips
ALLOWED_EXTS = {".mp3", ".wav", ".ogg", ".flac"}
ALLOWED_QUALITY = {"A", "B"}


def build_session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    s.params = {"key": api_key}  # type: ignore[assignment]
    return s


def length_to_seconds(s: str | None) -> float | None:
    """Parse XC length string 'M:SS' or 'H:MM:SS' into seconds."""
    if not s:
        return None
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def query_species(
    session: requests.Session,
    genus: str,
    species: str,
    rate_limit_s: float,
) -> list[dict[str, Any]]:
    """Fetch ALL pages for a gen/sp query. Returns a list of recordings."""
    page = 1
    out: list[dict[str, Any]] = []
    while True:
        params = {"query": f"gen:{genus} sp:{species}", "page": page}
        try:
            r = session.get(API_V3, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"    [warn] request error: {exc}")
            return out
        if r.status_code == 401:
            raise SystemExit(f"Unauthorized: {r.text[:200]}")
        if r.status_code == 429:
            print("    [warn] rate limited, sleeping 30s")
            time.sleep(30)
            continue
        if r.status_code != 200:
            print(f"    [warn] HTTP {r.status_code}: {r.text[:200]}")
            return out
        try:
            data = r.json()
        except json.JSONDecodeError:
            print("    [warn] non-JSON response")
            return out
        recs = data.get("recordings") or []
        out.extend(recs)
        num_pages = int(data.get("numPages") or 1)
        if page >= num_pages or not recs:
            break
        page += 1
        time.sleep(rate_limit_s)
    return out


def rank_recordings(recs: list[dict[str, Any]], max_seconds: int) -> list[dict[str, Any]]:
    """Filter by quality + duration + license presence, sort short-first."""
    filt = []
    for r in recs:
        q = (r.get("q") or "").upper()
        if q not in ALLOWED_QUALITY:
            continue
        if not r.get("lic"):
            continue  # must be properly licensed/attributed
        if not r.get("file"):
            continue
        filt.append(r)
    # Sort: <=max_seconds first (ascending), then longer (ascending)
    def key(r: dict[str, Any]) -> tuple[int, float]:
        sec = length_to_seconds(r.get("length")) or 1e9
        bucket = 0 if sec <= max_seconds else 1
        return (bucket, sec)
    filt.sort(key=key)
    return filt


def safe_ext_from_url(url: str) -> str:
    path = urlparse(url).path
    for ext in ALLOWED_EXTS:
        if path.lower().endswith(ext):
            return ext
    return ".mp3"  # XC default


def sanitize_id(xc_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(xc_id))


def download_file(session: requests.Session, url: str, dest: Path) -> int:
    """Stream a file with a 20 MB hard cap. Returns bytes written or -1 on error."""
    try:
        with session.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                return -1
            # Refuse HTML error pages
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "html" in ctype:
                return -1
            size = int(r.headers.get("Content-Length") or 0)
            if size and size > MAX_BYTES:
                return -1
            tmp = dest.with_suffix(dest.suffix + ".part")
            written = 0
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > MAX_BYTES:
                        tmp.unlink(missing_ok=True)
                        return -1
                    f.write(chunk)
            tmp.rename(dest)
            return written
    except requests.RequestException:
        return -1


def process_species(
    session: requests.Session,
    row: pd.Series,
    out_dir: Path,
    max_per_species: int,
    max_seconds: int,
    rate_limit_s: float,
    disk_budget_remaining: int,
) -> dict[str, Any]:
    sci = str(row["scientific_name"]).strip()
    parts = sci.split()
    if len(parts) < 2:
        return {"primary_label": row["primary_label"], "scientific_name": sci,
                "status": "skipped_bad_name", "n_hits": 0, "n_downloaded": 0, "bytes": 0}
    genus, species = parts[0], parts[1]
    species_dir = out_dir / str(row["primary_label"])
    species_dir.mkdir(parents=True, exist_ok=True)

    recs = query_species(session, genus, species, rate_limit_s)
    filtered = rank_recordings(recs, max_seconds)

    meta_path = species_dir / "metadata.csv"
    fieldnames = ["xc_id", "filename", "scientific_name", "gen", "sp", "en",
                  "length", "q", "lic", "rec", "cnt", "url", "also", "type"]
    meta_exists = meta_path.exists()
    meta_f = meta_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(meta_f, fieldnames=fieldnames)
    if not meta_exists:
        writer.writeheader()

    n_dl = 0
    bytes_used = 0
    try:
        for r in filtered:
            if n_dl >= max_per_species:
                break
            if bytes_used >= disk_budget_remaining:
                break
            xc_id = sanitize_id(r.get("id") or "")
            if not xc_id:
                continue
            ext = safe_ext_from_url(r.get("file") or "")
            fname = f"XC{xc_id}{ext}"
            dest = species_dir / fname
            if dest.exists():
                continue
            size = download_file(session, r.get("file"), dest)
            time.sleep(rate_limit_s)
            if size <= 0:
                continue
            n_dl += 1
            bytes_used += size
            writer.writerow({
                "xc_id": r.get("id"),
                "filename": fname,
                "scientific_name": f"{r.get('gen', '')} {r.get('sp', '')}".strip(),
                "gen": r.get("gen"),
                "sp": r.get("sp"),
                "en": r.get("en"),
                "length": r.get("length"),
                "q": r.get("q"),
                "lic": r.get("lic"),
                "rec": r.get("rec"),
                "cnt": r.get("cnt"),
                "url": r.get("url") or r.get("file"),
                "also": ";".join(r.get("also") or []) if isinstance(r.get("also"), list) else r.get("also"),
                "type": r.get("type"),
            })
    finally:
        meta_f.close()

    return {
        "primary_label": row["primary_label"],
        "scientific_name": sci,
        "status": "ok" if n_dl > 0 else ("no_matches" if not filtered else "no_downloads"),
        "n_hits": len(recs),
        "n_filtered": len(filtered),
        "n_downloaded": n_dl,
        "bytes": bytes_used,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-shot", type=Path,
                    default=Path("/home/soh/birdclef-2026/data/zero_shot_species.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/home/soh/birdclef-2026/data_external/xenocanto"))
    ap.add_argument("--max-per-species", type=int, default=20)
    ap.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    ap.add_argument("--rate-limit-s", type=float, default=1.0)
    ap.add_argument("--disk-budget-gb", type=float, default=5.0)
    ap.add_argument("--api-key", type=str, default=os.environ.get("XC_API_KEY"))
    args = ap.parse_args()

    if not args.api_key:
        print(
            "No XenoCanto API key found. API v2 was retired; v3 requires a key.\n"
            "Get one free at https://xeno-canto.org/account and re-run with\n"
            "    XC_API_KEY=<key> python scripts/download_xenocanto.py ...\n"
            "Exiting without downloading."
        )
        sys.exit(2)

    zs = pd.read_csv(args.zero_shot)
    birds = zs[zs["class_name"].str.strip().str.lower() == "aves"].reset_index(drop=True)
    print(f"Zero-shot birds to query: {len(birds)}")
    if len(birds) == 0:
        print("Nothing to do — no zero-shot bird species.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = build_session(args.api_key)
    inv_path = args.out_dir / "_inventory.csv"
    disk_budget = int(args.disk_budget_gb * (1024 ** 3))
    total_bytes = 0

    with inv_path.open("w", newline="", encoding="utf-8") as inv_f:
        inv_writer = csv.DictWriter(
            inv_f,
            fieldnames=["primary_label", "scientific_name", "status",
                        "n_hits", "n_filtered", "n_downloaded", "bytes"],
        )
        inv_writer.writeheader()
        for i, row in birds.iterrows():
            remaining = max(0, disk_budget - total_bytes)
            if remaining == 0:
                print(f"[{i+1}/{len(birds)}] disk budget exhausted; stopping")
                break
            print(f"[{i+1}/{len(birds)}] {row['primary_label']} ({row['scientific_name']})")
            res = process_species(
                session=session,
                row=row,
                out_dir=args.out_dir,
                max_per_species=args.max_per_species,
                max_seconds=args.max_seconds,
                rate_limit_s=args.rate_limit_s,
                disk_budget_remaining=remaining,
            )
            res.setdefault("n_filtered", 0)
            inv_writer.writerow(res)
            total_bytes += int(res.get("bytes") or 0)
            if (i + 1) % 20 == 0:
                print(f"  -- progress: {i+1}/{len(birds)} species, "
                      f"{total_bytes/1024/1024:.1f} MiB used --")
    print(f"Done. Total bytes: {total_bytes/1024/1024:.1f} MiB. Inventory: {inv_path}")


if __name__ == "__main__":
    main()
