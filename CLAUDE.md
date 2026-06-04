# BirdCLEF+ 2026

Kaggle competition: multi-taxa bioacoustic classification in the Pantanal.

## Quick Start

```bash
# Train baseline (EfficientNet-B0 SED)
python train.py --fold 0

# Train larger teacher (EfficientNet-B3)
python train.py --backbone tf_efficientnet_b3.ns_jft_in1k --fold 0 --batch_size 16 --experiment_name teacher_b3

# Generate submission
python inference.py --checkpoints checkpoints/baseline/best_fold0.pt --test_dir data/test_soundscapes
```

## Architecture

SED (Sound Event Detection) with:
- timm backbone (configurable: efficientnet_b0/b3, convnext, etc.)
- GEM frequency pooling (learnable p, interpolates avg↔max)
- Attention-weighted temporal pooling (class-specific)
- Asymmetric Loss (ASL) for long-tail multi-label

## Key Decisions

- ASL over BCE: 234 classes, >99% negative labels per sample → need asymmetric gradient budget
- SED over clip classification: preserves temporal evidence for overlap handling
- Differential LR: backbone at 0.1× to preserve pretrained features
- Inverse-sqrt class-balanced sampling for long-tail species

## Project Structure

- `src/config.py` — experiment configuration (dataclasses)
- `src/dataset.py` — audio loading, mel spectrograms, data splits
- `src/models.py` — SED model (backbone + GEM + attention)
- `src/losses.py` — ASL, focal, distillation losses
- `src/augmentations.py` — mixup, specaugment, gain
- `src/utils.py` — metrics, seeding
- `train.py` — training entry point
- `inference.py` — submission generation with post-processing
