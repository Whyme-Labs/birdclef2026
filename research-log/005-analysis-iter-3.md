# Analysis Iteration 3 — V183 broke 0.941 to 0.942: Real signal or noise?

**Date:** 2026-05-06
**Phase:** 5
**Iteration:** 3
**Status:** in-progress

## Context

After ~30 LB submissions, V183 (mean of 3 SSL probes: BirdMAE + AudioMAE + BioLingual, used as 5th member at weight 0.10) scored 0.942 — the first break above 0.941. V185 (added BirdAVES → 4 probes) and V186 (Bird-MAE FT classifier alone) failed to compound. V187 (3 probes + Bird-MAE FT averaged) is still scoring on Kaggle.

The user has asked: what direction would *actually* work to close the 0.014 gap to top LB 0.956? Apply the updated thinking frameworks.

## Strong Baseline Gate Audit (RETROACTIVE)

**Claim:** "V183 = 0.942 vs V137 = 0.941 → +0.001 break of the ceiling."

Audit:
- [ ] Is V137 a strong baseline? **Honest answer: Unclear.** V137 = our internal best ensemble, NOT current SOTA. Top LB is 0.956 — that's the real baseline. V137 is at 0.941, which is 0.015 below SOTA. **V137 fails the Strong Baseline Gate as a "thing to beat":** beating it by 0.001 may be noise; the relevant gap is +0.015.
- [ ] Has the noise floor been established? **No.** Kaggle public LB shows historical noise of ~0.001-0.002 between identical-or-near-identical submissions. **A single +0.001 result is at the noise floor** and cannot be claimed as a real signal without replication.

**Implication:** V183 = 0.942 may not be a real signal. We have been narrating a +0.001 as a "breakthrough" when it might be variance.

This is a **disconfirm signal we have been ignoring**. Per anti-fragility: take the disagreement seriously.

## Prediction Calibration Audit (RETROACTIVE)

What we predicted vs. what happened across the recent batch:

| Run | Predicted | Actual | Signal | Note |
|-----|-----------|--------|--------|------|
| V179 (V137-pseudo retrained V2S → replace) | beat-baseline (val 0.9474 → expect ≥0.941) | 0.938 | **disconfirm** | val_auc misled us; LB worse than V137 |
| V181 (XC data added → replace V2S) | beat-baseline (val 0.9793 → expect ≥0.942) | 0.939 | **disconfirm** | same pattern: val ≠ LB |
| V182 (Bird-MAE probes 5th @ 0.10) | beat-baseline (research said +10-16 mAP for Bird-MAE) | 0.941 | **disconfirm** | identity, no compound |
| V183 (3-probe stack 5th @ 0.10) | beat-baseline (~+0.002 expected) | 0.942 | **partial** | direction right, magnitude smaller than expected |
| V184 (3-stack at weight 0.15) | beat-baseline (more weight = more gain) | TIMEOUT | **null** | poor experiment design — not robust to inference budget |
| V185 (4-probe with BirdAVES) | beat-baseline (compound +0.001-0.002) | 0.942 | **disconfirm** | predicted compound, got identity |
| V186 (Bird-MAE FT classifier alone) | beat-baseline (val 0.9746 → expect 0.943-0.944) | 0.941 | **disconfirm** | val_auc-LB mismatch again |

**Pattern:** We have been *systematically over-predicting LB gains based on val_auc and "feature quality" reasoning.* The model of the problem implicit in our predictions is wrong.

## What the Disconfirms Are Telling Us (First Principles)

The single most informative finding: **our metric of model quality (val_auc on focal-recording-derived val) does NOT predict LB performance above 0.94.**

- V179 val 0.9474 → LB 0.938 (-0.003)
- V181 val 0.9793 → LB 0.939 (-0.002)
- V186 val 0.9746 → LB 0.941 (no gain)
- V137 V2S R2 val 0.9454 → LB 0.941

Higher val_auc has *negative or zero* correlation with LB above the threshold.

**Why?** Because val is focal recordings (close-mic, single-bird, isolated) and LB is soundscape (distant-mic, multi-bird, ambient noise). The val signal measures a different distribution than the LB signal.

**This is the load-bearing assumption that has been failing all our recent experiments.** When we re-rank candidate models by val_auc, we are filtering for *focal-recording quality*, not LB quality.

## Reframing: What is V137 Actually Missing?

Apply Occam's Razor: simplest explanation for why we are stuck at 0.941-0.942.

V137 = ensemble of 4 CNN/ViT models. Each predicts independently per 5s chunk. Errors are CORRELATED across the 4 models because:
- They were all trained on the same focal-recording distribution
- They all use mel spectrograms (similar input representation)
- They all use sigmoid + multi-label head (same output structure)

Adding a 5th member at weight 0.10 with another mel/sigmoid representation cannot break this — the dilution math caps it because the *correlation structure* doesn't change.

**The genuinely orthogonal information we are not using:**

1. **Cross-chunk temporal context.** V137 predicts each 5s chunk independently. But within a 60s recording, species presence is correlated across time. We have a per-recording matrix of 12 chunks × 234 species, but treat it as 12 independent predictions.

2. **V137's prediction structure itself.** When V137 is uncertain (low max prob, high entropy), it's wrong more often. We don't use this. A model that learns "when V137's prediction looks like X, the true label looks like Y" is genuinely different from another mel-based classifier.

3. **Per-recording *consistency* — soundscape segmentation.** A bird vocalizing in chunk 3 likely produces partial calls in chunks 2 and 4. V137's per-chunk independence misses this.

## Candidate New Direction (Phase 2 Hypothesis Draft)

**Hypothesis:** The 0.941-0.942 ceiling reflects independent per-chunk prediction. A *meta-learner* trained on V137's per-recording (12, 234) prediction matrix to predict refined chunk-level labels can break it, because it captures temporal/structural information V137 cannot.

**Concrete proposal:**
- Input: V137's per-recording (12, 234) prediction matrix (after QMix), per soundscape
- Target: per-chunk multi-hot labels (from labeled soundscape — 1478 labeled chunks across ~120 soundscapes)
- Architecture: small Transformer (4 layers, dim 256, heads 4) over the 12 chunk vectors
- Loss: BCE with class-balancing
- Output: corrected (12, 234) predictions

This is fundamentally different from probes because:
- Probes use *raw audio features* → 234 classes (independent per chunk)
- Meta-learner uses *V137's predictions* → 234 corrections (cross-chunk attention)

The meta-learner is trained on labeled soundscape directly, so it minimizes the distribution gap (focal val → soundscape LB) that has been killing us.

## Quantified Prediction (Required by Predict-Then-Run)

| Aspect | Prediction | Confidence | Rationale |
|---|---|---|---|
| Local val (on labeled soundscape held-out fold) | macro-AUC 0.78-0.85 | medium | similar to probe-only AUCs we observed (~0.75 zero-shot, ~0.85 trained) |
| Public LB delta when added as 5th @ 0.10 | +0.001 to +0.003 | low | this is genuinely uncertain — could be 0 (more correlated than we think) or +0.005 (real new info) |
| Public LB if it fully replaces 5th-member probes | +0.000 to +0.005 | low | replacing 5th member is constrained by calibration |

**Disconfirm condition:** if local val AUC < 0.65 (worse than zero-shot probes on labeled soundscape), the meta-learner is not extracting useful patterns from V137's outputs. Abandon and pivot.

**Confirm condition:** local val AUC > 0.85 AND LB delta > 0 (any positive). This would establish that meta-learning on V137 predictions adds real information.

## Anti-Stacking Check

Is this stacking?

- **Bad stacking:** "Add a meta-learner ON TOP OF probes ON TOP OF V137"
- **What this is:** A reframing — V137 is no longer treated as the final answer; it is treated as a *feature extractor for soundscape distribution*. The meta-learner explicitly models the cross-chunk structure that V137 ignores.

The novelty claim: **V137 as soundscape-distribution feature extractor, not as a classifier.** This is a conceptual shift, not a mechanical addition.

## What This Replaces in the Pipeline

Currently:
```
Audio → V137 (Perch+SED+NFNet+V2S) → QMix → final probs
       ↓ 5th member adds (V162/V174/V182/V183)
```

Proposed:
```
Audio → V137 (Perch+SED+NFNet+V2S) → QMix → V137 (12,234)
                                            ↓ Meta-learner (transformer over chunks)
                                            ↓
                                            Refined (12,234) → final probs
```

The meta-learner is a *post-V137 refiner* trained on the labeled-soundscape distribution. It is not a 5th ensemble member; it is a structural correction.

## Decision

Recommend: **Phase 2 (formalize hypothesis with mathematical justification)** before any code is written. Specifically:

1. Verify V183 = 0.942 is real signal (not noise) by running V183 again with a single line changed (e.g., random seed for any stochastic step) — if reproducible, real. If 0.941 second time, noise.

2. Generate V137-on-labeled-soundscape predictions matrix as input features for the meta-learner (12, 234) per soundscape.

3. Probe whether the meta-learner can fit at all (PoC): tiny MLP, 50 train chunks, see if val AUC > 0.5.

4. If PoC works, scale to full transformer architecture.

## Next Steps

- Submit V184 timeout root cause investigation (Kaggle 9h hit) — was it the inference cost of the heavier-weighted stacked probes?
- Set up a held-out labeled-soundscape fold for the meta-learner experiments (this is the immutable evaluation contract for this iteration)
- Draft the Phase 2 hypothesis document (with proper math)
- Estimate compute: meta-learner training is ~30 minutes; V137-prediction-matrix generation is the compute bottleneck (3-5h)

## Unanswered Questions for User

1. Is V183 = 0.942 a finding worth re-testing for noise? (Costs 1 quota slot; high information value.)
2. Are we comfortable abandoning the "more probes" direction in favor of "meta-learner on V137 predictions"?
3. Realistic stretch goal: 0.945 (+0.003 from current) by competition end, or do we aim for top-3?
