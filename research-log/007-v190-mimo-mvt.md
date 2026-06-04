# V190 MVT: MiMo encoder pre-RVQ embeddings as bird-audio probe

**Date:** 2026-05-02
**Phase:** 5
**Iteration:** 4
**Status:** completed

## Context

After the V188 disconfirm — labeled-soundscape macro-AUC has zero gradient toward LB because V137's probes were trained on it — we accepted that local AUC cannot pre-rank candidates above 0.94. The user redirected to MiMo-Audio-Tokenizer (Xiaomi, 2025): a 1.2B Transformer trained at 24kHz with LM-loss-driven RVQ codebooks. Three options identified:
1. **Frozen pre-RVQ probe (this run)** — does the encoder transfer to bird audio at all?
2. WavTokenizer drop-in test — alternative neural audio tokenizer (~1d)
3. Train a domain-adapted classification-gradient VQ-VAE on bird audio (~3-5d)

Option 1 is the gate for option 3.

## Hypothesis

MiMo's encoder, trained on speech + general audio, learns acoustic primitives (transients, pitched segments, noise textures) that should partially overlap with the structure of bird vocalizations. A frozen 1280-dim mean-pooled embedding fed to per-species LR should distinguish species better than chance. If the speech-domain encoder specializes too narrowly to phonetics, AUC will be near-random.

## Prediction (Predict-Then-Run)

| Aspect | Prediction | Confidence |
|---|---|---|
| Held-out macro-AUC (5-fold GroupKFold) | 0.65–0.80 | medium |
| Direction relative to bird-specialty probes | weaker (Bird-MAE ~0.85) | medium |
| LB-transfer signal | **inconclusive from this MVT alone** | — |

**Rationale:** A 1280-dim feature vector from a 32-layer Transformer is high-capacity. Even a mismatched-domain encoder typically resolves species above 0.65 by virtue of capturing amplitude/spectral statistics. Conversely, speech tokenizers compress phonetic structure, which is partly orthogonal to bird call structure (no formants, narrowband sweeps), so we shouldn't expect Bird-MAE-level AUCs.

## Critical caveat

Per V188 disconfirm: macro-AUC on labeled soundscape **does not predict LB performance**. This MVT *only* tests for the presence of any signal. AUC ≥ 0.80 does not imply MiMo would help on Kaggle test; AUC < 0.65 does imply MiMo is unlikely to help anywhere.

Use the result as a **gate** on option 3 (3-5 day classification-gradient VQ-VAE), not as a green light to push V137+MiMo to LB.

## Decision rules

- AUC < 0.65 → ABANDON MiMo direction. Speech encoder fails to transfer; option 3 not justified.
- AUC ∈ [0.65, 0.80) → AMBIGUOUS. Domain-adapted MiMo might work but other directions likely higher EV.
- AUC ≥ 0.80 → option 3 (classification-gradient VQ-VAE on bird audio) is justified investment.

## Anti-stacking check

Is this stacking?
- **Bad:** "Add MiMo probe as 5th-of-5 on top of existing 3-probe stack."
- **What this is:** A *capability test* of a fundamentally different feature space (speech tokenizer pre-RVQ embeddings, never trained on natural sound). Not yet committed to ensembling. Used to gate a larger-investment experiment (option 3).

## Engineering notes

- **flash-attn blocker:** MiMo's attention uses `flash_attn_varlen_func` which requires SM 7.5+. The local 1080 Ti is SM 6.1. Workaround: shim `flash_attn` module in `sys.modules` with a `torch.scaled_dot_product_attention` fallback that handles the variable-length packing layout. Identical-length batches (our 5s chunks) hit the fast path; variable-length falls back to a per-sequence loop.
- **Disk pressure:** local FS is 98% full. MiMo tokenizer = 3.7GB safetensors. Downloading drops free space to ~7GB; tight but feasible.
- **Pascal precision:** no native bf16. Forcing fp32 on GPU.

## Strong baseline gate (retroactive audit)

The "baseline" V190 is testing against is "any non-trivial feature extractor produces signal on labeled soundscape." Concretely:
- V189 PCEN+KMeans+LR (untrained, hand-coded) achieved AUC 0.7255.
- Bird-MAE/AudioMAE/BioLingual probes (specialty bird models) ~0.80–0.90.

So the strong baseline for "MiMo encoder transfers to bird audio" is "AUC at least matches the no-training V189 baseline (0.72)". An AUC < 0.72 means MiMo with all its 1.2B params does *worse* than KMeans on PCEN — which would be a strong disconfirm.

Updating the decision threshold:
- AUC < 0.72 → MiMo loses to KMeans → speech encoder is anti-correlated with bird structure (strong disconfirm)
- AUC ∈ [0.72, 0.85) → MiMo has signal but no advantage over specialty encoders we already use
- AUC ≥ 0.85 → MiMo has *meaningfully different* signal than our existing probes (interesting)

## Result (2026-05-02 14:18 UTC)

**Mean macro AUC: 0.8283 ± 0.0171** across 5-fold GroupKFold by recording.

| Fold | AUC | n_classes |
|---|---|---|
| 0 | 0.8292 | 28 |
| 1 | 0.8418 | 30 |
| 2 | 0.8024 | 30 |
| 3 | 0.8176 | 32 |
| 4 | 0.8504 | 27 |

Total runtime: ~90s feature extraction + ~5s LR fitting on a 1080 Ti via SDPA fallback. 739 chunks × 1280-d features.

## Prediction vs. Reality

Predicted: 0.65–0.80 macro AUC (medium confidence). Observed: 0.8283.

**Signal: partial-confirm.** Direction was correct (signal exists). Magnitude was slightly above my predicted upper bound — MiMo's speech-derived encoder transfers *better* than I expected to bird audio. This is a third consecutive over-prediction-on-the-low-side surprise (V189 PCEN got 0.72 vs predicted 0.55-0.70 ceiling).

**Calibration insight:** I have been systematically underestimating how much signal mid-quality feature extractors carry on labeled soundscape. The 1280-d feature space + per-class LR is more discriminative than the "speech ≠ bird" prior suggested.

**Anti-fragility tracker update:**

| Run | Predicted | Observed | Signal |
|-----|-----------|----------|--------|
| V189 | local 0.55-0.70 | 0.7255 | partial (over-low) |
| V190 | local 0.65-0.80 | 0.8283 | partial (over-low) |

I systematically *underestimate* local AUC of mid-quality feature extractors and *overestimate* LB gains from any single-probe addition. Both biases stem from the same root: I'm using LB-scale prior as my noise floor for local AUC, when local labeled-soundscape distribution is much narrower than LB.

## Decision

The V190 strong-baseline gate said:
- AUC < 0.72 → strong disconfirm (anti-correlation with bird structure)
- AUC ∈ [0.72, 0.85) → has signal, **no advantage over existing probes**
- AUC ≥ 0.85 → meaningfully different signal

V190 = **0.83**, which falls into "has signal but not better than existing probes." MiMo encoder is comparable to AudioMAE/BioLingual probes (~0.80-0.85), not better.

**Implication for option 3:** A 3-5 day classification-gradient VQ-VAE would aim to produce features whose AUC > 0.88 (significant lift over current probes). Going from MiMo's 0.83 to 0.88+ via domain adaptation is plausible but not guaranteed. Cost-benefit weakens.

**Better near-term path:** Test MiMo features as a 4th probe in the V183 stack. The probe is computed; the engineering cost is low (modify the V183 stack-mean kernel to include a 4th source). Only LB submission resolves whether it adds beyond the 3-probe stack.

## Operational constraint: Kaggle CPU budget

MiMo encoder = ~640M params (32 × d_model 1280 × FFN 5120). On Kaggle 4-core CPU at ~0.5 TFLOPS:
- ~1 TFLOP per chunk × 8400 chunks = **~16,800 seconds (4.6h)** for MiMo alone

V137 takes ~5h on Kaggle CPU. V137 + MiMo would land near 9.5–10h — over the 9h hard limit. V184 (3-stack at higher weight) and V187 (4-source) already timed out at this margin.

**This blocks V191 = "add MiMo to V183 stack" without compute reduction:**
- ONNX int8 quantization (4-8x speedup) — 1+ day engineering, untested on MiMo
- Strided extraction (process every 2nd/3rd chunk, interpolate) — degrades signal
- Distill MiMo into smaller student — multi-day GPU job

## Revised decision

V190 confirms MiMo encoder transfers (0.83 AUC, comparable to AudioMAE/BioLingual, below Bird-MAE-Base). But the path from "MiMo features have signal" to "MiMo features improve LB" is blocked operationally by Kaggle's 9h CPU budget.

**Higher-EV moves with current LB quota:**

1. **V191 = V183 reconfirmation submission.** Re-push V183 kernel with a seed-perturbed inference order. If LB = 0.942 → 0.001 break is real. If 0.941 → V183 was a noise sample, V137 ceiling is unbroken. **Cost:** 1 LB slot, **info:** binary on the only positive signal we have. **EV: very high.**

2. **V192 = V183 stack at weight 0.07 or 0.13 (sweep 0.10).** Tests whether 5th-member weight is sub-optimal. Engineering: trivial (1-line change). **EV: medium.**

3. **V193 = post-V137 per-recording calibration.** Within each 60s recording, compute per-class chunk-distribution shape and apply soft normalization (e.g., subtract median chunk score to remove per-recording drift). Tests whether per-recording bias is unmodeled. Engineering: ~30 min. **EV: medium-low** (per-recording drift may already be absorbed by QMix rank-ensemble).

4. **MiMo direction parked.** Only revisit if V191 confirms 0.942 is real AND we have GPU time to distill MiMo encoder into Kaggle-friendly size.

## Next steps

1. Update memory with V190 finding [done]
2. Schedule V191 = V183 reconfirmation as the next LB submission (highest EV per slot)
3. Park MiMo direction; option 3 (gradient-trained VQ-VAE) does NOT clear cost-benefit given Kaggle budget block

## Anti-fragility tracker

Recent prediction track record (calibration check):

| Run | Predicted | Observed | Signal |
|-----|-----------|----------|--------|
| V179 | LB ≥ 0.941 | 0.938 | disconfirm |
| V181 | LB ≥ 0.942 | 0.939 | disconfirm |
| V182 | LB > 0.941 | 0.941 | disconfirm |
| V183 | LB +0.002 | +0.001 (0.942) | partial |
| V185 | LB +0.001 (compound) | 0.942 | disconfirm |
| V186 | LB 0.943-0.944 | 0.941 | disconfirm |
| V189 | local 0.55-0.70 | 0.7255 | partial (wrong upper) |

We over-predict on ~70% of runs. V190's 0.65-0.80 prediction should be interpreted with this bias — observed could easily be lower. If observed AUC > 0.80, that's surprising upside given our calibration drift.
