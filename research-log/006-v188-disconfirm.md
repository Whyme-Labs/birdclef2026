# V188 disconfirm: Labeled soundscape is contaminated eval

**Date:** 2026-05-06
**Phase:** 5
**Iteration:** 3
**Status:** completed
**Signal:** disconfirm (informative)

## Prediction vs. reality

Predicted (medium confidence): held-out macro-AUC 0.85 ± 0.03 — meta-learner adds modest correction on top of V137.

Observed: V137 macro-AUC = **1.0000** on every held-out fold. Meta-learner delta: **+0.0000**. Cross-recording GroupKFold did not change this.

## What this teaches us

The eval signal is broken at its source. V137's per-class probes were trained on the ENTIRE labeled-soundscape pool (~1478 chunks across ~120 recordings). When we hold out 20% of recordings for eval, V137's predictions on those held-out recordings are still ~perfect because:

1. The probe is trained per-class across all recordings, not per-recording
2. Labeled-soundscape distribution is narrow enough that held-out recordings look like training recordings
3. The probes already saw similar acoustic conditions during training

V137 generalizes within the labeled-soundscape *distribution* essentially perfectly (1.0 AUC), but only generalizes to the hidden Kaggle test at 0.941. **The 0.059 gap is the actual generalization signal we need to attack.** Our local eval cannot see this gap.

## Load-bearing assumption that just broke

We assumed: "if we can improve macro-AUC on labeled soundscape, we improve LB." This assumption has been quietly load-bearing for V162/V174/V182/V183 probe selection, V185 BirdAVES probe, and V188 meta-learner design.

V188 disconfirms it directly: on labeled soundscape V137 = 1.0, on LB V137 = 0.941. **Labeled-soundscape AUC has zero gradient toward LB.**

This is consistent with the earlier disconfirms:
- V179 val 0.9474 → LB 0.938
- V181 val 0.9793 → LB 0.939
- V186 val 0.9746 → LB 0.941

Every signal we've been using to pre-rank candidate experiments has been wrong. The only honest signal is Kaggle LB itself.

## Implication

The realistic remaining path:
- Accept that local eval cannot pre-filter promising directions
- Use Kaggle quota strategically: each submission has high marginal information value
- Restructure: instead of "training-eval-LB three-stage funnel," go direct from theory to LB submission for each new direction
- Token MVT (V189, running) is exempt because it tests a *different* claim (presence of signal in tokens), not generalization

## Next steps

1. Kill V188 meta-learner direction (will not work with current eval)
2. Wait for V189 token MVT result — definitive answer on token signal
3. Save remaining quota (today: 2/5 unused) for the highest-information submissions:
   - **Best use:** re-submit V183 with a different random seed inside the kernel to confirm 0.942 is signal, not noise. If 0.942 reproduces → real. If 0.941, then V183 was a single noise sample.
   - **Better use:** test a genuinely new direction once we have V189 result

## Decision

PIVOT. Move from "design experiments with local eval guidance" to "predict-then-LB-test." Each LB submission is a full experiment now.
