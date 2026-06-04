# BirdCLEF 2026 External Data Acquisition Summary

Date: 2026-04-17
Current LB: 0.941 (target: break the ceiling with external data)

## TL;DR

- **Downloaded: BirdCLEF 2025 training audio (7.2 GB, 28,564 recordings, 206 species)**
  via user-uploaded mirror `samrohrer/birdclef-minus-humans` (human voice stripped)
- **42 species (18% of 2026's 234 species) have direct scientific-name overlap with 2025**
- **13,481 extra labeled recordings immediately usable — +38% training data**, ~2.1x
  average boost for each of those 42 species
- **BirdCLEF 2021-2024 competition downloads all returned HTTP 403 Forbidden**
  (rules not yet accepted; per instructions, competition-rules acceptance was NOT
  performed). Metadata-only acquired from public mirrors.
- All disk-writes kept well under the 80 GB budget (currently 7.4 GB used, 41 GB free)

## What was downloaded

| Path | Source | Size | Content |
|------|--------|------|---------|
| `data_external/2025_audio/birdclef-2025/train_audio/` | `samrohrer/birdclef-minus-humans` | 7.2 GB | 206 species × 28,564 OGG recordings, human voice stripped |
| `data_external/2025_audio/birdclef-2025/{taxonomy,train,sample_submission}.csv` | same | 5 MB | 2025 species list + per-recording metadata |
| `data_external/extra_2025/` | `aayush26/extra-birdclef-dataset25-audio-format` | 46 MB | Supplemental XC recordings of 2025-competition species (bc00/bc20/bc21/bc23/bc24 organized dirs). Only adds 63 files for species overlapping with 2026 — marginal |
| `data_external/meta/` | various | 92 MB | Metadata-only CSVs for BirdCLEF 2021–2024 (below) |
| `data_external/species_lists/` | `samvelkoch/{bird-clef-2024,birdclef-2025}-species-images` | < 1 MB | Extracted scientific-name lists per year (taxonomy cross-reference only) |
| `data_external/external_data_overlap.csv` | computed | 2 MB | Canonical mapping: source_year, source_path, scientific_name, 2026 primary_label, available flag |

### Metadata-only files (no audio)

Located in `data_external/meta/`:

| File | Year | Content |
|------|------|---------|
| `train_ext_2021.csv` | 2021 | 62,874 recordings × 397 species (scientific_name, code, filename, secondary_labels, lat/lon, etc.) |
| `train_ext.csv` | 2022 | 14,852 recordings × 152 species |
| `train_meta_audio_durations.csv` | 2023 | 16,941 recordings × 264 species |
| `cleaned_metadata.csv` | 2024 | ~24k unique recordings × 183 species (from cleaned BirdCLEF-2024) |
| `train_extended.csv` | 2020 (XC) | Xeno-Canto extended for 264 species (Birdsong Recognition 2020) — 4 overlap with 2026 |
| `all_train.csv` | mix | Cross-year primary_label index (seshurajup) |
| `train_21_22_23.csv`, `train_23.csv` | 21-23 | Ollypowell cleaned labels |

## Species overlap table

Overlap matched by lowercased `scientific_name` against 2026 `taxonomy.csv`.

| Year | Total species | Total recs | Overlap species | Overlap recs | Status |
|------|--------------|-----------|-----------------|--------------|--------|
| **2025** | 206 | 28,564 | **42** | **13,481** | DOWNLOADED (via mirror) |
| 2021 | 397 | 62,874 | 34 | 7,380 | audio BLOCKED (403) |
| 2022 | 152 | 14,852 | 5 | 974 | audio BLOCKED (403) |
| 2024 | 183 | 24,451 | 1 | 500 | audio BLOCKED (403) |
| 2023 | 264 | 16,941 | 1 | 119 | audio BLOCKED (403) |
| 2020 XC | 264 | 23,784 | 4 | 1,355 | available (`rohanrao/xeno-canto-bird-recordings-extended-{a-m,n-z}`, 29 GB total) — not downloaded because mostly duplicates of 2025 overlap |

### Why 2025 is the jackpot

BirdCLEF 2025 was the Colombia El Silencio / Pantanal-adjacent competition — it shares the Neotropical biome with BirdCLEF 2026 (Pantanal, Brazil), so most of its 206 species are taxonomically close. 42 of them have direct scientific-name matches.

Sample of high-value overlap species (recordings: 2026 current → after merge):

| 2026 code | Species | 2026 | +2025 | Ratio |
|-----------|---------|------|-------|-------|
| grekis | Pitangus sulphuratus | 482 | 1,472 | 3.1x |
| compau | Nyctidromus albicollis | 493 | 1,301 | 2.6x |
| trokin | Tyrannus melancholicus | 483 | 1,270 | 2.6x |
| roahaw | Rupornis magnirostris | 489 | 1,198 | 2.4x |
| banana | Coereba flaveola | 498 | 1,108 | 2.2x |
| … 37 more … | | | | |
| 67252 | Trachycephalus typhonius | 6 | 20 | 3.3x |
| 41970 | Panthera onca | 21 | 36 | 1.7x |

**Aggregate for the 42 overlap species: 12,381 → 25,862 recordings (2.1x)**.
Total 2026 training set grows from 35,549 → 49,030 (+38 %).

## Coverage gap

- 179 / 234 (76 %) 2026 species have NO external recordings from any past BirdCLEF:
  - 110 Aves (mostly new Pantanal endemics)
  - 33 Amphibia (frogs — past BirdCLEFs were bird-only)
  - 28 Insecta (cicadas etc. — same)
  - 7 Mammalia, 1 Reptilia
- These classes (Amphibia / Insecta / Mammalia / Reptilia) are intrinsically unreachable
  from BirdCLEF 2021-2025 since those competitions only contained birds.

For the non-overlap species we must rely on the existing iNaturalist/XC recordings
already present in `data/train_audio` (2026's own source), pseudo-labels, and possibly
iNaturalist direct API scraping (already set up in `data_external/inaturalist/` — removed
as it was empty).

## Competition rules status (what's blocked)

All five `kaggle competitions download -c birdclef-{2021..2025}` requests return
`403 Forbidden`. The Kaggle CLI offers no `accept-rules` subcommand — acceptance
requires signing in to the competition's web page and clicking "Late Submission"
or similar. Per task instructions, this was **not** performed.

If rules are later accepted, the following volumes become available:
- 2021: ~75 GB competition archive (34 species overlap, +2,620 extra recordings NOT in 2025)
- 2022: ~6 GB (only 5 species overlap, 974 recs — likely duplicated via XC)
- 2023: ~13 GB (1 species overlap, 119 recs — skip)
- 2024: ~26 GB (1 species overlap, 500 recs — skip)
- 2025 official: ~12 GB (same content as the downloaded mirror)

## Recommended training strategy

### Phase 1 (immediate)

1. **Merge 2025 audio into training loop** with a `source_year` tag.
   - In `src/dataset.py`, extend the training DataFrame to include rows from
     `data_external/2025_audio/birdclef-2025/train_audio/{primary_label}/*.ogg`
     re-labeled with the 2026 primary_label via `external_data_overlap.csv`.
   - Keep the 2025 recordings **only for the 42 overlap species** (13,481 clips).
2. **Weight external samples lower**. Start with `sample_weight = 0.5` on 2025
   recordings — they're from a slightly different soundscape (El Silencio vs. Pantanal)
   and may contain different background noise. Sweep 0.25 / 0.5 / 1.0.
3. **Do NOT add to validation set.** Keep validation on the same 2026 soundscape
   fold splits to preserve LB-comparable local CV.

### Phase 2 (if LB responds positively)

4. If Phase 1 shows +0.005 or better on LB, accept 2021 competition rules manually
   and download the 11 species (2,620 recordings) that 2025 does NOT have:
   `Glaucidium brasilianum, Myiarchus tyrannulus, Pandion haliaetus, Passer
   domesticus, Piaya cayana, Ramphocelus carbo, Saltator coerulescens,
   Sittasomus griseicapillus, Synallaxis albescens, Thamnophilus doliatus,
   Tringa melanoleuca`.
5. Consider MixUp / CutMix across external + internal data to improve
   generalization to Pantanal soundscape.

### Phase 3 (longer shot)

6. For the 76 % non-overlap species, consider:
   - iNaturalist audio scraping (API) — esp. for the Insecta/Amphibia classes
   - Fine-tuning Perch v2 / AudioMAE backbones on the 2025 audio as an auxiliary
     pretraining step (domain adaptation), then fine-tune on 2026.

### What NOT to do

- Don't mix in 2024 audio if obtained later — only 1 species overlap, high
  distribution shift (S. Asian birds).
- Don't mix in 2022/2023 audio — negligible overlap.
- Don't evaluate local CV against merged data — will inflate numbers.

## Files to consume in training code

- `/home/soh/birdclef-2026/data_external/external_data_overlap.csv` — drive the merge
- `/home/soh/birdclef-2026/data_external/2025_audio/birdclef-2025/train_audio/` — audio
- `/home/soh/birdclef-2026/data_external/2025_audio/birdclef-2025/train.csv` — 2025 labels
- `/home/soh/birdclef-2026/data_external/2025_audio/birdclef-2025/taxonomy.csv` — 2025 species

Disk usage of `data_external/` total: 7.4 GB. Free disk: 41 GB.
