# BirdCLEF+ 2026 — Experiment Log

## Competition Overview

- **Task**: Predict species presence probabilities per 5-second window in Pantanal soundscapes
- **Metric**: Macro-averaged ROC-AUC (skips classes with no positives)
- **Species**: 234 (162 birds, 35 amphibians, 28 insects, 8 mammals, 1 reptile)
- **Constraint**: CPU-only notebook, ~90 min runtime
- **Deadline**: 2026-06-03
- **Public LB**: Top is 0.926, public blend baselines at 0.87-0.89

## Data Summary

| Dataset | Size | Description |
|---------|------|-------------|
| Train recordings | 35,549 files (11GB) | XenoCanto (23K) + iNaturalist (12.5K), 206/234 species |
| Labeled soundscapes | 1,478 windows / 66 files | Pantanal, multi-label (avg 4.2 species/window), all 234 species |
| Unlabeled soundscapes | 10,658 files (5.1GB) | Pantanal target domain — used for pseudo-labeling |
| Test soundscapes | Hidden | Used by Kaggle for scoring |

**Critical finding**: 28 species have ZERO training recordings. All 28 appear in the labeled soundscapes. 25 are sonotaxon variants of species 47158, 3 are regular species (25073, 517063, 1491113).

## Architecture

### SED Model (Sound Event Detection)

```
Input mel spectrogram (3, 224, 313)
    ↓
EfficientNet backbone (pretrained, ImageNet)
    ↓
Feature maps (C, 7, 10)
    ↓
GEM Frequency Pooling (learnable p, init=3.0)
    → GEM(x; p) = (1/F · Σ_f x_f^p)^{1/p}
    ↓
Frame features (C, 10)
    ↓
Attention Temporal Pooling (class-specific)
    → α_t = softmax_t(tanh(W_a · h_t))
    → ŷ = Σ_t α_t ⊙ (W_c · h_t)
    ↓
Clip predictions (234,) + Frame predictions (10, 234)
```

**Why SED over clip classification**: BirdCLEF soundscapes are polyphonic — multiple species overlap temporally. Class-specific attention lets different species attend to different time frames, recovering localization information that a global pool destroys.

### Loss: Asymmetric Loss (ASL)

```
L_ASL = -(1/C) Σ_c [ y_c · (1-p_c)^γ+ · log(p_c) + (1-y_c) · p̂_c^γ- · log(1-p̂_c) ]
```

Where `p̂_c = max(p_c - m, 0)` is the shifted probability for negatives.

- `γ+ = 0`: No focusing on positives (every positive matters)
- `γ- = 4`: Strong down-weighting of easy negatives
- `m = 0.05`: Probability shift creates dead zone for confident negatives

**Why ASL over BCE**: With 234 classes and typically 1-3 positives per sample, >99% of labels are negative. ASL shifts the gradient budget toward rare positive signals.

### Augmentations

- **Mixup** (α=0.5, prob=0.5): Simulates polyphonic soundscapes by mixing spectrograms and soft labels
- **SpecAugment**: 2 frequency masks (max 20 bins) + 2 time masks (max 30 frames)
- **Gain augmentation**: Random volume scaling (0.8-1.2x)

### Training Details

- Optimizer: AdamW with differential LR (backbone 0.1×, head 1×)
- Scheduler: Cosine annealing with linear warmup
- Mixed precision (AMP) for ~2× throughput
- Class-balanced sampling via inverse-sqrt weighting

## Experiments Run

### Experiment 1: B0 Baseline

**Goal**: Establish baseline SED model on labeled data + labeled soundscapes.

| Parameter | Value |
|-----------|-------|
| Backbone | tf_efficientnet_b0.ns_jft_in1k (4.6M params) |
| Training data | 28,439 recordings + 1,478 soundscape windows |
| Batch size | 32 |
| LR | 1e-3 (backbone 1e-4) |
| Epochs | 30 |
| Time/epoch | ~371s |

**Result: val_auc = 0.9489 (epoch 20)**

Training curve:
```
Epoch  1: 0.5926 (warmup)
Epoch  3: 0.8702
Epoch  6: 0.9356
Epoch 12: 0.9487
Epoch 20: 0.9489 ← best
Epoch 30: 0.9474
```

**Observations**:
- Fast convergence — 0.93+ by epoch 6
- Plateau around 0.948-0.949 after epoch 12
- Including labeled soundscape windows was critical for the 28 zero-shot species

### Experiment 2: B3 Teacher

**Goal**: Train larger model for stronger pseudo-labeling and teacher signal.

| Parameter | Value |
|-----------|-------|
| Backbone | tf_efficientnet_b3.ns_jft_in1k (11.4M params) |
| Training data | 28,439 recordings + 1,478 soundscape windows |
| Batch size | 16 |
| LR | 5e-4 (backbone 5e-5) |
| Epochs | 30 |
| Time/epoch | ~690s |

**Result: val_auc = 0.9496 (epoch 16)**

Training curve:
```
Epoch  1: 0.5930
Epoch  4: 0.9111 (vs B0: 0.9075)
Epoch  8: 0.9436 (vs B0: 0.9393)
Epoch 16: 0.9496 ← best (vs B0: 0.9489)
Epoch 30: 0.9471
```

**Observations**:
- B3 consistently 0.003-0.004 ahead of B0 at equivalent epochs
- Final improvement over B0 is small (+0.0007) — diminishing returns from backbone scaling alone
- Confirms the strategy doc's point: "just fine-tune one model harder" won't win

### Experiment 3: Pseudo-Labeling (Round 1)

**Goal**: Generate target-domain pseudo-labels from B0+B3 teacher ensemble.

| Parameter | Value |
|-----------|-------|
| Teachers | B0 baseline + B3 teacher (equal weight average) |
| Data | 10,658 unlabeled Pantanal soundscapes |
| Filtering | Per-class adaptive threshold: θ_c = clip(μ_c + 1.0·σ_c, 0.2, 0.9) |
| Labels | Soft (preserve teacher uncertainty) |

**Result: 127,793 pseudo-labeled windows, 201 species covered**

- 99.9% of predictions passed sample-level filter (min_max_prob ≥ 0.2)
- Threshold range: [0.200, 0.746]
- Top species: 22973 (13,385), 517063 (12,944), 23158 (10,147) — dominated by amphibians/insects
- 33 species not detected (very rare or absent from soundscapes)

### Experiment 4: B0 Student with Pseudo-Labels

**Goal**: Retrain B0 on labeled + pseudo-labeled data for domain adaptation.

| Parameter | Value |
|-----------|-------|
| Backbone | tf_efficientnet_b0.ns_jft_in1k (4.6M params) |
| Training data | 28,439 recordings + 1,478 soundscapes + 127,793 pseudo-labeled (157,710 total) |
| Pseudo-label weight | 0.5 (half contribution to loss) |
| Batch size | 32 |
| LR | 5e-4 (backbone 5e-5) |
| Epochs | 20 |
| Time/epoch | ~1845s (~5.3× longer than baseline) |

**Result: val_auc = 0.9393 (epoch 20)**

Training curve:
```
Epoch  1: 0.7094 (high loss from soft pseudo-labels)
Epoch  3: 0.9084
Epoch  9: 0.9340
Epoch 11: 0.9381
Epoch 16: 0.9387
Epoch 19: 0.9391
Epoch 20: 0.9393 ← best
```

**Key insight**: Val AUC (0.9393) is LOWER than baseline (0.9489) on clean recordings. This is expected and desirable — the validation set consists of focal recordings, not soundscapes. The pseudo-label model is adapting to the target domain (soundscapes), trading clean-recording accuracy for soundscape robustness. The real test is the LB (soundscape-based).

**Scientific basis**: This matches Lasseck (BirdCLEF 2024): pseudo-labeling on target-location recordings significantly improved competition performance despite not improving on source-domain validation.

## Final Submission

### 3-Model Blend (Kaggle Notebook v7)

| Model | Weight | Role |
|-------|--------|------|
| B0 baseline | 0.25 | Clean-data specialist |
| B3 teacher | 0.35 | Stronger representation |
| B0 pseudo-R1 | 0.40 | Domain-adapted for soundscapes |

**Post-processing**:
1. File-level max prior: `p̃(t) = p(t) + 0.05 · max_t' p(t')` — species persistence assumption
2. Confidence-sharpened smoothing: sharpen(κ=1.5) → Gaussian smooth(σ=0.7) → unsharpen — preserves peaks while removing noise
3. No probability clipping — AUC handles >1.0 fine, clipping creates rank ties

**Kaggle assets**:
- Dataset: `whymelabs/birdclef2026-sed-b0-baseline` (3 checkpoints, 80MB total)
- Notebook: `whymelabs/birdclef2026-sed-submission` (v7, CPU mode)
- Submit at: https://www.kaggle.com/code/whymelabs/birdclef2026-sed-submission

## Project Structure

```
birdclef-2026/
├── src/
│   ├── config.py          # Experiment configuration (dataclasses)
│   ├── dataset.py         # BirdCLEFDataset, SoundscapeDataset, PseudoLabelDataset
│   ├── models.py          # SEDModel (backbone + GEM + AttentionHead)
│   ├── losses.py          # AsymmetricLoss, FocalLoss, DistillationLoss
│   ├── augmentations.py   # Mixup, SpecAugment, GainAugment
│   └── utils.py           # macro_auc, set_seed, AverageMeter
├── train.py               # Training entry point
├── inference.py            # Submission generation with post-processing
├── pseudo_label.py         # Pseudo-labeling pipeline
├── kaggle_notebook/        # Kaggle submission notebook
├── kaggle_model/           # Model checkpoints for Kaggle upload
├── checkpoints/            # Local checkpoints (3 experiments)
├── pseudo_labels/round1/   # 127K pseudo-labeled windows
├── data/                   # Competition data (16GB)
└── references/             # Downloaded public notebooks
```

## What Worked

1. **SED architecture** (vs clip classifier): Attention pooling with GEM gave strong baseline
2. **ASL loss**: Clear improvement over BCE for extreme multi-label imbalance
3. **Including labeled soundscapes in training**: Critical for 28 zero-shot species
4. **Pseudo-labeling on target soundscapes**: 127K windows of domain-adapted training data
5. **Soft pseudo-labels**: Preserving teacher uncertainty prevents error amplification
6. **3-model blend with representation diversity**: B0 + B3 + domain-adapted B0

## Phase 2: Adopting Public Perch-finetuned Models

### Critical Discovery: Architecture & Pretraining Mismatch

Our V1 models (ImageNet-pretrained, Linear attention head) scored 0.853 on LB — far below the 0.89 public baseline.

**Root causes identified**:
1. **Wrong architecture**: V1 head uses `Linear` for att/cls. Public baseline uses `FC(Linear→ReLU→Dropout) + Conv1d`. State dict keys differ.
2. **Wrong pretraining**: ImageNet features are fundamentally inferior to Perch bioacoustic features for audio tasks.

**Fix**: Created `src/models_v2.py` matching public V2 architecture exactly, downloaded Perch-finetuned checkpoints (LB862.pt + LB872.pt from tonylica/birdclef-2026-model).

### Experiment 5: V2 Architecture with Perch Checkpoints (V10)

| Config | Value |
|--------|-------|
| Models | LB862 (Perch baseline) + LB872 (Perch finetuned) |
| Blend | 0.8 LB872 + 0.2 LB862 (probability space) |
| Post-processing | File-max prior + confidence-sharpened smoothing |

**Result: LB = 0.890** ← matches public baseline!

### Experiment 6: Fine-tuning Perch Models on Labeled Data

Fine-tuned both LB862 and LB872 on labeled recordings + labeled soundscapes.

| Model | Epochs | LR | Val AUC |
|-------|--------|----|---------|
| LB872 finetuned | 10 | 1e-4 (backbone 5e-6) | 0.9709 |
| LB862 finetuned | 10 | 1e-4 (backbone 5e-6) | 0.9707 |

### Experiment 7: 4-Model Ensemble (V11)

| Model | Weight |
|-------|--------|
| LB862 (Perch) | 0.15 |
| LB872 (Perch) | 0.15 |
| Finetuned LB862 | 0.35 |
| Finetuned LB872 | 0.35 |

**Result: LB = 0.882** ← WORSE than V10 (0.890)!

**Key insight**: Fine-tuning on labeled recordings degrades soundscape performance. The Perch models were calibrated for soundscape-style inputs. Our finetuning pulled them toward focal recording characteristics, hurting generalization.

### Experiment 8: V2 Pseudo-Labeling

Generated pseudo-labels using LB862+LB872 as teachers on 10,658 soundscapes.
- 127,896 predictions → 85,427 retained (66.8%)
- Threshold range: [0.300, 0.586]

Retrained finetuned LB872 with pseudo-labels:
- Best val_auc = 0.9675 (vs 0.9709 without pseudo-labels)
- Not submitted — expected to perform even worse on LB given Exp 7 findings

### Experiment 9: TTA with Time-Shifts (V12)

Reverted to proven LB862+LB872 blend. Added 3× TTA with time-shift offsets (0, +1.25s, -1.25s).
- **Result: LB = pending**

### Experiment 10: No Post-Processing Ablation (V13)

Clean 2-model blend (LB862+LB872, 0.8/0.2) without any post-processing.
- **Result: LB = pending**

### Model Soup Experiment

Averaged weights of LB862 and LB872 (identical architecture). Evaluated on labeled soundscapes:
- LB862: soundscape AUC = 0.9972
- LB872: soundscape AUC = 0.9984
- Soup: soundscape AUC = 0.9980
- Conclusion: marginal, not worth a separate submission

## Submission History

| Version | Description | LB Score |
|---------|-------------|----------|
| V7 | 3-model blend (B0+B3+pseudo) with post-processing | 0.853 |
| V8 | B0+B3 only (no pseudo) with post-processing | 0.836 |
| V10 | V2 architecture, LB862+LB872 Perch, 0.8/0.2 blend | **0.890** |
| V11 | 4-model (Perch + finetuned), weighted blend | 0.882 |
| V12 | LB862+LB872 + TTA (3 time-shifts) | pending |
| V13 | LB862+LB872, no post-processing | pending |

## Key Learnings

1. **Perch pretraining >> ImageNet for bioacoustics**: This is the single biggest insight. Perch-finetuned B0 (0.890) crushes ImageNet-pretrained B3 (0.836).
2. **Don't finetune soundscape-optimized models on recordings**: Fine-tuning on focal recordings degrades soundscape performance (0.890 → 0.882).
3. **Architecture matters**: V2 head (FC+Conv1d) vs V1 (Linear) makes a meaningful difference.
4. **Ensemble diversity needs different pretraining, not just finetuning**: Adding finetuned variants of the same base models doesn't add real diversity.

## What Could Improve

1. **Different backbone architectures** (EfficientNet-B3, EfficientNetV2-S, ConvNeXt) with Perch-style pretraining
2. **External data from Xeno-Canto** and historical BirdCLEF years (top-5 teams all used this)
3. **OpenVINO fp16 conversion** for faster CPU inference (fit more models/TTA in time budget)
4. **Multi-round pseudo-labeling** training new models from scratch (not finetuning existing ones)
5. **Soundscape-only finetuning**: Only use labeled soundscapes (not recordings) to preserve soundscape calibration
6. **Per-class calibration**: Post-hoc Platt scaling on OOF predictions
7. **ONNX export** for faster CPU inference
