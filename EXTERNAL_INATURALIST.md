# iNaturalist Zero-Shot Amphibian Audio Pull

Date: 2026-04-17
Source: iNaturalist public API (https://api.inaturalist.org/v1)
Download script: `/home/soh/birdclef-2026/download_inat.py`
Output dir: `/home/soh/birdclef-2026/data_external/inaturalist/<primary_label>/`

## Goal

Pull up to 50 recordings per species for 3 Pantanal amphibians with zero training data
in BirdCLEF 2026.

## Per-species results

| primary_label | scientific_name       | taxon_id | obs with has_sound=true | obs with populated `sounds[]` | downloaded | bytes |
| ------------- | --------------------- | -------- | ----------------------- | ------------------------------ | ---------- | ----- |
| 1491113       | Adenomera guarani     | 1491113  | 1                       | 0                              | 0          | 0     |
| 25073         | Chiasmocleis mehelyi  | 25073    | 4                       | 0                              | 0          | 0     |
| 517063        | Pithecopus azureus    | 517063   | 206                     | 1                              | 1          | 20040 |

The given `primary_label` values ARE the canonical iNaturalist taxon IDs for these
species (verified by scientific-name lookup; `Phyllomedusa azurea` resolves to the
same id 517063 — it's a synonym for `Pithecopus azureus`).

## Total disk usage

- Files on disk: 1 (`.m4a`)
- Total bytes: 20,040 (≈ 20 KB)
- `/home/soh/birdclef-2026/data_external/inaturalist/` block usage: 40K

## Downloaded files

```
data_external/inaturalist/517063/obs70002751_s0.m4a   (19.6 KB, m4a, 2.14 s @ 44.1 kHz)
```

License: CC-BY, Marcos Severgnini. Source file URL:
`https://static.inaturalist.org/sounds/165988.m4a`

## Notes on species without recordings

For Adenomera guarani (1 obs) and Chiasmocleis mehelyi (4 obs) the iNaturalist search
index marks these observations as `has_sound=true`, but when fetched directly (both
through `/v1/observations` and individual `/v1/observations/<id>`) every `sounds`
array is empty. These are index false positives — the observations in fact have
photos only (`observation_sounds_count == 0` in the legacy endpoint). There is no
publicly-accessible audio for these two species on iNaturalist.

For Pithecopus azureus the mismatch is the same: the index reports 206 obs with
sound, but scanning all 206 only finds 1 observation (id 70002751) that has a real
sound file attached. The other 205 are photos-only false positives.

## Recommendation

iNaturalist is not a useful audio source for these three species. Alternative
external sources to try for zero-shot amphibian recordings:

- Fonozoo (Spanish Animal Sound Library) — strong for South American amphibians
- AmphibiaWeb / Dendrobatidae.org
- Macaulay Library (Cornell) — has some anuran calls
- Xeno-canto — already scraped, known empty for these taxa
- Brazilian academic collections: Coleção Bioacústica da Unesp Rio Claro, Museu de
  Zoologia USP (often require direct contact but host Pantanal frog recordings)

## Script parameters

- Rate limit: 1.05 s between API requests
- Per-species cap: 50 recordings
- File size cap: 30 MB per file (skipped if exceeded)
- Minimum valid file size: 2 KB
- Endpoint: `GET /v1/observations?taxon_id=<id>&has_sound=true&per_page=200`
- Summary JSON written to `data_external/inaturalist/_summary.json`
