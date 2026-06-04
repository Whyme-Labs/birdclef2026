# User idea inventory + next-build plan

**Date:** 2026-05-06
**Phase:** 5
**Iteration:** 6
**Status:** in-progress

## What the user proposed and outcome

| User idea | Status | Result |
|---|---|---|
| MiMo audio tokenizer | tested | V190 = 0.83 AUC (transfers); operationally blocked on Kaggle (640M-param encoder, 4.6 h CPU); ONNX FP32 export works; INT8 broken on var-length packing |
| MiMo richer engineering | tested | V190b mean+max+std = 0.832 (+0.004); MLP probe = 0.815 (overfits). Pooling tricks at the limit |
| Sparse motif tokens (PCEN+KMeans) | tested | V189 = 0.7255; partial signal, not better than continuous |
| JEPA family (BirdAVES) | tested | V185 = 0.942 LB (no compound); ortho proves anti-additive |
| JEPA family (WavJEPA-Nat) | tested | 0.5415 single AUC; **best k=5 with BirdMAE+AudioMAE+BioLingual+MiMo+WavJEPA = 0.6096**; Kaggle inference too slow (7.6 s/chunk → 17.7 h vs 9 h budget) |
| "Signals conflicting or repeating" | tested | Mean off-diag corr 0.40 (diverse); BirdAVES anti-additive; AudioMAE-BirdMAE most redundant (0.50) |
| Multi-iter Noisy Student | NOT STARTED | The 2024-2025 winning recipe; multi-day GPU job; **highest-EV unstarted move** |
| QMix α sweep | NOT STARTED | Single-line change in V137; tests rank-vs-prob blend ratio; cheap |
| Cross-recording temporal aggregation | NOT STARTED | Treats 12 chunks/recording as bag, not independent; pure post-V137 reasoning, no training |
| Replace V137 base component | NOT STARTED | E.g., Conformer or SSAST replacing V2S; multi-day; structural |

## What we have NOT tried (still on the table)

1. **Multi-iter Noisy Student (V194)** — the canonical winning recipe. Cycle: V137 pseudo-labels train+XC+soundscape → train fresh student → student becomes new V2S. Repeat until converged. Multi-day GPU but background-runnable. **My recommendation as primary next big build.**

2. **QMix α sweep** — V137 uses α=0.5 (50/50 prob/rank blend). Try α=0.4, α=0.6 as one-line changes. Each test costs 1 LB slot. EV: low (single number tweak) but cheap.

3. **Cross-recording temporal pooling** — within each 60s recording, soft-aggregate the 12 chunk-level predictions: chunks where a species is loud should pull up neighbors where it's barely audible. Pure post-V137 logic, no training, ~1 day kernel engineering. **EV: medium** if per-recording structure was being thrown away.

4. **WavJEPA distillation** — train a 50M-class student on WavJEPA's 768-d targets. Once distilled, student fits Kaggle budget and unlocks the 5-probe stack (orthogonality says 0.6096 vs V137's 0.604). 1-2 days GPU.

5. **MiMo encoder rewrite for static shapes** — kill the variable-length packing inside the encoder forward (we always feed fixed 5s × 24 kHz mel, T=501). After this, MiMo ONNX INT8 should not break, giving 3-4× speedup. ~half-day engineering.

## Plan: parallel builds while waiting for slot reset

The LB constraint binds tightly: we have 1 slot left today (V191 already submitted, waiting for score). Tomorrow: 5 fresh slots.

**Now (parallel):**
- **Multi-iter NS V194** — start training round 1 on local GPU. Can run for hours/days in background.
- **V192 Path B (WavJEPA stride 3)** — ~30 min engineering. Stage for tomorrow's first slot.
- **MiMo static-shape rewrite** — staged for after V194 round 1 settles.

**Today's last slot (if used):**
- Either burn it now on a Path B V192 (risky — may still timeout)
- Or save it for tomorrow's higher-confidence experiment

**Tomorrow's slots (5 fresh):**
- Slot 1: V192 Path B (4-probe with WavJEPA stride-3)
- Slot 2: QMix α sweep (V137 α=0.4)
- Slot 3: V194 round 1 NS (replaces V137 V2S with iter-1 student)
- Slot 4-5: reserve for the highest-EV finding from slots 1-3
