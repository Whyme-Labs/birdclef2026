# BirdCLEF+ 2026 Research Setup

**Date:** 2026-05-06
**Phase:** 0
**Iteration:** 1
**Status:** completed

## Context

Multi-taxa bioacoustic classification competition (Pantanal, Brazil). 234 species across Aves, Amphibia, Insecta, Mammalia, Reptilia. Macro-AUC scoring on 5-second test soundscape chunks.

We have been working in this competition for weeks; this log retroactively captures the research state as we reached our first break above the 0.941 ceiling at V183 = 0.942.

## Idea DNA

**Problem (bedrock):** V137 ensemble has plateaued at LB 0.941 across 25+ single-component variations. Top LB 0.956 implies a 0.014 gap that cannot be closed by single 5th-member additions or single-component swaps.

  - First-principles audit: 0.941 is not a fundamental ceiling of the data — it's the ceiling of the *4-model rank ensemble at QMix(α=0.5) with 5th member at weight 0.10*. The structure, not the data, is the constraint.

**Assumption (inferred):** When a single new probe at weight 0.10 is added to V137, the rank-ensemble dilution math saturates regardless of feature quality. But averaging multiple probes from *orthogonal feature spaces* reduces error variance in the stacked output; the 5th-member then carries lower-noise signal at the same effective weight.

  - Empirically validated by V183 = 0.942 (3-probe stack vs 0.941 individual probes).

**Novelty claim:** *Probe-space averaging as 5th-member error reduction* — not a new model, not a new training recipe, but a reframing of how to spend a fixed 5th-member weight budget. The unit of contribution is no longer "one feature space" but "a low-variance estimate aggregated across feature spaces".

## Domain

ML / audio classification / multi-label / soundscape / domain transfer (focal → soundscape).

## Success criteria

- **Primary metric:** public LB macro-AUC on Pantanal hidden test set
- **Threshold to claim success:** > 0.945 (current 0.942, top 0.956)
- **Stretch:** > 0.95 (within striking distance of top LB)

## Compute

- Local: GTX 1080 Ti (11GB VRAM), 30GB RAM, ~440GB disk (often >95% full)
- Kaggle: CPU-only inference, 9h hard limit, 5 submissions/day quota
- Constraints: 3-day GPU jobs feasible; Kaggle pipeline must finish in <5h to be safe with 9h budget

## Evaluation contract

**Mutable:** model code, architectures, hyperparameters, training data composition, ensemble weights, post-processing.

**Immutable (read-only):**
- Kaggle public/private LB scoring (cannot be modified)
- 234-species taxonomy (data/taxonomy.csv)
- Submission format: per-(file, end_time_5s_chunk) → 234 species probabilities
- Labeled soundscape val protocol (data/train_soundscapes_labels.csv) — used for probe training

**Primary metric:** Public LB macro-AUC. Local val_auc on focal recordings is **disqualified as a decision criterion** (verified across multiple experiments: local val_auc and LB are weakly correlated above 0.94).

## Research intensity

**Deep** — publication-grade investigation. We've already cited Rauch TMLR 08/2025 (Bird-MAE), BirdCLEF 2024-2025 winners, BirdAVES (Earth Species), AVES, BioLingual, AudioMAE, WavJEPA papers.
