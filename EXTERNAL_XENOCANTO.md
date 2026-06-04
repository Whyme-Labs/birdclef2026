# External XenoCanto acquisition — zero-shot species

Generated 2026-04-21 by `scripts/download_xenocanto.py` and
`scripts/identify_zero_shot.py`.

## Headline result

**0 recordings downloaded, 0 bytes used, 0 zero-shot birds.**

Every one of the 162 bird species (`class_name = Aves`) in
`data/taxonomy.csv` already has at least one entry in `data/train.csv`.
The 28 species that are zero-shot in the current training set are **all
non-birds** — 25 unnamed insect sonotypes plus 3 amphibians — and XenoCanto
is a bird-only archive, so the tool cannot fill those gaps.

## Step 1 — Zero-shot species (`data/zero_shot_species.csv`)

| Class    | Zero-shot | Total in taxonomy |
|----------|-----------|-------------------|
| Aves     | **0**     | 162               |
| Amphibia | 3         | 35                |
| Insecta  | 25        | 28                |
| Mammalia | 0         | 8                 |
| Reptilia | 0         | 1                 |
| **Total**| **28**    | **234**           |

Zero-shot rows (full list, none queryable on XC):

| primary_label | scientific_name      | class_name |
|---------------|----------------------|------------|
| 1491113       | Adenomera guarani    | Amphibia   |
| 25073         | Chiasmocleis mehelyi | Amphibia   |
| 517063        | Pithecopus azureus   | Amphibia   |
| 47158son01 – 47158son25 | Insect son01…son25 | Insecta |

The three amphibians have real Linnaean names but are still outside the
XenoCanto scope. The 25 insect entries are anonymised sonotypes with no
genus/species resolution, so they can't be looked up anywhere.

## Step 2 — XenoCanto API availability

- `GET https://xeno-canto.org/api/2/recordings?query=gen:Hirundo&page=1`
  → **HTTP 404** with body
  `"Xeno-canto API v2 is no longer available. Visit https://xeno-canto.org/explore/api for API v3 documentation."`
- `GET https://xeno-canto.org/api/3/recordings?query=gen:Hirundo`
  → **HTTP 401** with body
  `"Missing or invalid 'key' parameter. Visit https://xeno-canto.org/account to retrieve your API key."`
- The browser-facing `/explore/api` docs page is protected by Anubis bot
  mitigation and can't be fetched headlessly.

The spec's example (v2, no key) therefore can no longer work. v3 requires
per-account authentication. No such key is present in this environment
(checked `env`, `~/.netrc`, project tree — none found).

## Step 3 — Downloads

Skipped, because:

1. **No zero-shot birds** (the only class XenoCanto covers).
2. No v3 API key is available in this environment.

Condition (1) alone is sufficient to stop — even with a working key the
zero-shot list contains nothing a bird archive could satisfy.

## Step 4 — Inventory

```
data_external/xenocanto/
└── (empty — no recordings written)
```

Disk usage: **0 B** of the 5 GB budget.
Still zero-shot after this step (all 28, unchanged):

- 3 amphibians: `1491113` Adenomera guarani, `25073` Chiasmocleis mehelyi,
  `517063` Pithecopus azureus
- 25 insect sonotypes: `47158son01` … `47158son25`

## Follow-ups if you need coverage for these 28 species

XenoCanto is the wrong corpus. Useful alternatives:

- **iNaturalist research-grade observations** with audio for the three
  amphibians (you already ingest iNat — filter by `inat_taxon_id`
  `1491113`, `25073`, `517063` from the global taxon dump).
  `train.csv` currently has 0 iNat rows for those taxa, but a broader
  sweep of the iNat audio dump may surface some.
- **BDVA / AmphibiaWeb / Neotropical anuran archives** for the frogs.
- **Orthoptera Species File (OSF) / SINA** or the
  [Global Cicada Sound Collection] for cicada-like insects — matching
  `47158son01…25` would require the BirdCLEF organiser's mapping from
  sonotype code back to a genus; without that, these remain unidentifiable.
- **Ask the competition host** for the sonotype→taxon map so the 25
  insect entries can be de-anonymised.

## Artefacts produced

| Path | Purpose |
|------|---------|
| `scripts/identify_zero_shot.py`   | Writes `data/zero_shot_species.csv` |
| `data/zero_shot_species.csv`      | 28 species with no rows in `train.csv` |
| `scripts/download_xenocanto.py`   | XC v3 downloader; key-aware, runs automatically when a bird gap exists and `XC_API_KEY` is set |
| `data_external/xenocanto/`        | Target directory (empty) |
| `EXTERNAL_XENOCANTO.md`           | This report |

The downloader is ready to run should a future zero-shot list include
Aves entries:

```bash
XC_API_KEY=<your-key> \
  /home/soh/miniconda3/envs/birdclef/bin/python \
  /home/soh/birdclef-2026/scripts/download_xenocanto.py
```

It enforces the original constraints: quality A/B only, up to 20
recordings per species, ≤60 s preferred, ≤20 MB per file, 5 GB global
budget, 1 s rate limit, attribution (license + recorder) persisted to
`<primary_label>/metadata.csv`, and a global `_inventory.csv` with hit
counts per species.
