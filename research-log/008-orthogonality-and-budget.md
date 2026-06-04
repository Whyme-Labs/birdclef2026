# Orthogonality finding + Kaggle compute as structural ceiling

**Date:** 2026-05-06
**Phase:** 5
**Iteration:** 5
**Status:** completed

## Context

User pushed back on "all my ideas (JEPA, tokenisation) not working?" — pointed out we hadn't quantified probe redundancy, MiMo had more pooling-engineering room, and WavJEPA-Nat had never actually been probed. Three parallel investigations:

1. **Probe orthogonality analysis** (BirdMAE, AudioMAE, BioLingual, BirdAVES, MiMo, WavJEPA on the same 739 labeled-soundscape chunks, 5-fold GroupKFold, OOF predictions)
2. **MiMo richer pooling** (V190b: mean+max+std + MLP probe)
3. **WavJEPA-Nat extraction** + add to ortho

## Results

### Probe orthogonality (6-probe, OOF, 64 valid classes)

```
Per-probe AUC:        Best K trajectory:
AudioMAE   0.6040 ←  k=1: 0.6040  AudioMAE
BirdMAE    0.5688    k=2: 0.6049  +BirdMAE
MiMo       0.5596    k=3: 0.6081  +MiMo
BioLingual 0.5515    k=4: 0.6092  +BioLingual
WavJEPA    0.5415    k=5: 0.6096  +WavJEPA  ← OPTIMAL (drop BirdAVES)
BirdAVES   0.5143    k=6: 0.6069  +BirdAVES (DROPS)
```

Mean off-diagonal pairwise correlation: 0.40 (genuinely diverse). BirdAVES is anti-additive in every multi-probe configuration tested AND grid-search learned weights zero it out. AudioMAE-BirdMAE pair has highest correlation (0.50) — most redundant.

This explains V185's 0.942 plateau: V185 added BirdAVES which is the wrong probe. The right swap is BirdAVES → WavJEPA.

### MiMo richer pooling (V190b)

| Variant | AUC | Δ from V190 |
|---|---|---|
| Mean only (1280-d, LR) | 0.8283 | baseline |
| Mean+max+std (3840-d, LR) | 0.8323 | +0.004 |
| Mean+max+std (3840-d, MLP) | 0.8147 | -0.014 (overfits) |

MiMo's individual-AUC ceiling is ~0.83 with frozen-feature pooling tricks. MLP overfits with only ~1500 samples. Bigger gains require multi-layer features or attention pooling — not pursued today.

### WavJEPA extraction

200M-param Audio JEPA on raw waveform, 16kHz binaural. Local extraction on 1080 Ti: 43s for 739 chunks. Mean over channels and tokens → 768-d. Single-probe AUC 0.5415 (worst of 6) but **stack contributor at k=4**.

## V192 design + the Kaggle compute blocker

V192 = V185 with BirdAVES → WavJEPA (4-probe: BirdMAE + AudioMAE + BioLingual + WavJEPA). Pushed across 4 iterations:

- **v1:** WavJEPA failed via `trust_remote_code` cache bug (missing utils.py from cache copy when dataset name contains hyphens).
- **v2:** Manual `importlib.util.spec_from_file_location` import bypass works. All 4 probes ran on Kaggle dummy. **WavJEPA via PyTorch ≈ 7.6 s/chunk on Kaggle CPU.**
- **v3:** Refactored to ONNX path. Local export needed `nn.TransformerEncoderLayer.forward` monkey-patch (fused-op `aten::_transformer_encoder_layer_fwd` not ONNX-exportable). Validated 4e-6 relative match. Local ONNX FP32: 604 ms/chunk on i7-8700K. INT8 quantization HURTS (0.67x speedup, mixed quant/dequant overhead). Kernel had a syntax error from the if/elif/else refactor.
- **v4:** Syntax fix. ONNX path on Kaggle CPU: still ~7.6 s/chunk (Kaggle CPU 12x slower than my local desktop, not 5x). Other probes: <1 s/chunk.

**Budget projection for full LB run:** WavJEPA alone = 8400 × 7.6 s = 17.7 h. Plus V137 base (~5 h) = 22 h vs the 9 h Kaggle hard limit. **V192 will timeout if submitted as-is.**

### Honest synthesis

The orthogonality finding (drop BirdAVES, add WavJEPA) is real and actionable. But operationalizing it on Kaggle requires either:

1. **Subsample WavJEPA** (process every 3rd chunk, interpolate). Halves time → ~5.9 h. Fits with V137. Risk: ~0.001 AUC calibration loss.
2. **Distill WavJEPA** into a smaller student. Multi-day GPU work.
3. **Replace WavJEPA with a 100M-class waveform model** that fits budget naturally. We don't have one available.

## Strategic implication: the 0.014 gap

V137 = 0.941, top LB = 0.956. Gap = 0.014. Probe-stacking math (orthogonality data):
- V183 3-probe = 0.942 measured
- V192 4-probe (with WavJEPA) projected: +0.001 → 0.943
- Optimal 5-probe (with MiMo + WavJEPA) projected: +0.001 → 0.944

**The probe-stacking ceiling appears to be ~0.945.** Adding more probes at the saturated 5th-member-weight=0.10 position cannot close the gap to 0.956. The remaining +0.011 must come from a different mechanism:

1. **Multi-iter Noisy Student** — retrain V137's V2S component on V137 ensemble pseudo-labels, iterate until convergence. Standard 2024-2025 winning recipe. Multi-day GPU.
2. **Replace a V137 base component** (Perch / SED / NFNet / V2S) with a fundamentally different backbone. Not addition — substitution.
3. **Test-time augmentation / cross-recording aggregation** — operate on V137 outputs at inference time without retraining.

## Decision

Stop chasing probe stacking depth. The orthogonality finding is captured; V192 stays parked until ONNX speedup or a smaller waveform-domain probe arrives. Pivot in parallel to multi-iter NS as the primary ceiling-break attempt.

## Next steps

1. V191 LB result (pending) — confirms or refutes V183 = 0.942 noise.
2. Start V194 = multi-iter NS prep: pseudo-label generation on Xeno-Canto with V137 + first iter EffNetV2-S retrain. Multi-day local GPU job.
3. V192 path B (subsample WavJEPA stride 3) staged for tomorrow's quota IF V191 confirms direction.
