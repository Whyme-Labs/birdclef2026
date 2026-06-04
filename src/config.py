"""
Experiment configuration via dataclasses.

Design rationale:
- Dataclasses over YAML: type safety, IDE completion, no parsing overhead.
- All hyperparameters in one place for reproducibility.
- Differential LR for backbone vs head is critical: pretrained backbone needs
  gentler updates (typically 0.1x) to preserve learned representations.
"""
from dataclasses import dataclass, field
from typing import Optional
import argparse


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "data"
    # CSV with training metadata. Relative paths are resolved against data_dir.
    # Use "train_merged.csv" to train on 2026 + external 2025 data.
    train_csv: str = "train.csv"
    # Optional: base directory containing external 2025 train_audio.
    # Rows with source=="2025" in the merged CSV resolve their filenames here.
    external_2025_audio_dir: str = "data_external/2025_audio/birdclef-2025/train_audio"
    sample_rate: int = 32000        # Standard for bioacoustic work
    duration: float = 5.0           # Competition window size
    n_mels: int = 224               # Mel bins — matches backbone input height
    n_fft: int = 2048               # ~64ms window at 32kHz
    hop_length: int = 512           # ~16ms hop → 313 frames per 5s
    fmin: float = 0.0
    fmax: float = 16000.0           # Nyquist at 32kHz

    # ── Model ─────────────────────────────────────────────────────────────
    backbone: str = "tf_efficientnet_b0.ns_jft_in1k"
    num_classes: int = 234
    in_chans: int = 3               # Repeat mono mel to 3ch for pretrained
    gem_p_init: float = 3.0         # GEM init: p=1 → avg, p→∞ → max
    gem_p_trainable: bool = True
    drop_rate: float = 0.2

    # ── Training ──────────────────────────────────────────────────────────
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    backbone_lr_mult: float = 0.1   # Backbone gets lr * this
    weight_decay: float = 1e-5
    warmup_epochs: int = 3
    num_workers: int = 8
    grad_accum_steps: int = 1
    amp: bool = True
    max_grad_norm: float = 1.0

    # ── Loss ──────────────────────────────────────────────────────────────
    # ASL (Asymmetric Loss) — the key loss for long-tail multi-label.
    # Standard BCE is symmetric: it penalizes FP and FN equally. In BirdCLEF
    # where >99% of labels are negative, easy negatives dominate gradients.
    # ASL fixes this with:
    #   1. Probability shifting: p_m = max(p - m, 0) for negatives
    #   2. Asymmetric focusing: γ+ < γ- (harder on easy negatives)
    loss_type: str = "asl"          # 'asl', 'focal', 'bce'
    gamma_pos: float = 0.0          # Focusing for positives
    gamma_neg: float = 4.0          # Focusing for negatives (>> gamma_pos)
    clip_margin: float = 0.05       # Probability shift margin for negatives

    # ── Augmentation ──────────────────────────────────────────────────────
    mixup_alpha: float = 0.5        # Beta(α,α) — lower = less mixing
    mixup_prob: float = 0.5
    spec_augment: bool = True
    freq_mask_param: int = 20       # Max freq bins to mask
    time_mask_param: int = 30       # Max time frames to mask
    num_freq_masks: int = 2
    num_time_masks: int = 2

    # ── Cross-validation ──────────────────────────────────────────────────
    n_folds: int = 5
    fold: int = 0
    seed: int = 42

    # ── Distillation ──────────────────────────────────────────────────────
    # KD loss: α·L_hard(z_S, y) + (1-α)·τ²·KL(σ(z_T/τ) ‖ σ(z_S/τ))
    teacher_checkpoint: str = ""
    teacher_weight: float = 0.0     # 0 = no distillation
    temperature: float = 3.0

    # ── Resume ────────────────────────────────────────────────────────────
    resume_from: str = ""           # Path to checkpoint to warm-start model weights from
    resume_epoch: int = 0           # Skip N epochs (epoch counter starts here)

    # ── Pseudo-labeling ───────────────────────────────────────────────────
    pseudo_label_dir: str = ""      # Path to pseudo-label CSV
    pseudo_label_weight: float = 0.5  # Down-weight pseudo-labeled samples

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = "checkpoints"
    experiment_name: str = "baseline"

    @property
    def target_samples(self) -> int:
        return int(self.sample_rate * self.duration)

    @property
    def n_time_frames(self) -> int:
        """Number of mel spectrogram time frames for a full duration clip."""
        return (self.target_samples // self.hop_length) + 1


def config_from_args(args=None) -> Config:
    """Build Config from command-line args, using dataclass defaults."""
    parser = argparse.ArgumentParser()
    cfg = Config()
    for name, val in vars(cfg).items():
        if name.startswith("_"):
            continue
        ty = type(val) if not isinstance(val, bool) else lambda x: x.lower() in ("true", "1", "yes")
        parser.add_argument(f"--{name}", type=ty, default=val)
    parsed = parser.parse_args(args)
    return Config(**vars(parsed))
