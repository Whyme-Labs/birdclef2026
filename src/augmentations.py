"""
Data augmentations for bioacoustic spectrogram classification.

Each augmentation has a clear mathematical/empirical justification:

1. Mixup (Zhang et al., ICLR 2018):
   x̃ = λ·x₁ + (1-λ)·x₂,  ỹ = λ·y₁ + (1-λ)·y₂,  λ ~ Beta(α, α)

   For BirdCLEF, mixup in spectrogram space naturally simulates polyphonic
   soundscapes. Since test soundscapes contain multiple overlapping species,
   training with mixup creates a better distributional match.

   The 2025 2nd-place BirdCLEF+ paper explicitly credits MixUp as critical
   because it mimics the multi-label nature of test data.

2. SpecAugment (Park et al., Interspeech 2019):
   - Frequency masking: mask f consecutive mel bins, f ~ U[0, F]
   - Time masking: mask t consecutive time frames, t ~ U[0, T]

   Originally for ASR, but equally effective for bioacoustics. Acts as
   structured dropout that forces the model to not rely on any single
   frequency band or time segment. This improves robustness to partial
   occlusion (e.g., overlapping species, noise).

3. Gain augmentation:
   x̃ = x · g,  g ~ U[g_min, g_max]

   Simulates recording volume variation. Recordings in the training set come
   from diverse microphones and distances. This helps the model become
   invariant to absolute amplitude.
"""
import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """
    SpecAugment: frequency and time masking on mel spectrograms.

    Zeroes out random contiguous bands along frequency and time axes.
    Applied to the normalized spectrogram (after dB + min-max).
    """
    def __init__(self, freq_mask_param=20, time_mask_param=30,
                 num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, mel):
        """
        Args:
            mel: (C, F, T) mel spectrogram
        """
        _, n_freq, n_time = mel.shape

        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(n_freq - f, 1), (1,)).item()
            mel[:, f0:f0 + f, :] = 0.0

        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            t0 = torch.randint(0, max(n_time - t, 1), (1,)).item()
            mel[:, :, t0:t0 + t] = 0.0

        return mel


class Mixup:
    """
    Mixup augmentation at the batch level.

    Applied AFTER individual sample augmentations, operates on batched
    (mel, label) pairs. Returns mixed samples and soft labels.

    For multi-label: ỹ = λ·y₁ + (1-λ)·y₂ naturally produces valid
    soft multi-label targets (each class probability is in [0,1]).
    This is compatible with BCE/ASL losses without modification.
    """
    def __init__(self, alpha=0.5, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, mel, labels):
        """
        Args:
            mel: (B, C, F, T) batch of spectrograms
            labels: (B, num_classes) multi-hot labels
        Returns:
            mixed_mel, mixed_labels
        """
        if torch.rand(1).item() > self.prob:
            return mel, labels

        batch_size = mel.size(0)
        # Sample mixing coefficient: λ ~ Beta(α, α)
        # Beta(α, α) is symmetric around 0.5. Smaller α → more extreme mixing.
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        lam = max(lam.item(), 1 - lam.item())  # Ensure λ ≥ 0.5 (keep dominant)

        # Random permutation for pairing
        perm = torch.randperm(batch_size, device=mel.device)

        mixed_mel = lam * mel + (1 - lam) * mel[perm]
        mixed_labels = lam * labels + (1 - lam) * labels[perm]

        return mixed_mel, mixed_labels


class GainAugment(nn.Module):
    """
    Random gain augmentation in spectrogram space.

    Multiplies the spectrogram by a random factor, simulating volume
    variation across recordings. Applied before normalization in some
    pipelines, but here applied to normalized spectrograms as a
    scaling perturbation.
    """
    def __init__(self, min_gain=0.8, max_gain=1.2):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, mel):
        gain = torch.empty(1).uniform_(self.min_gain, self.max_gain).item()
        return (mel * gain).clamp(0, 1)


class TrainAugmentations(nn.Module):
    """Composed augmentation pipeline for training."""
    def __init__(self, config):
        super().__init__()
        self.spec_augment = SpecAugment(
            freq_mask_param=config.freq_mask_param,
            time_mask_param=config.time_mask_param,
            num_freq_masks=config.num_freq_masks,
            num_time_masks=config.num_time_masks,
        ) if config.spec_augment else None
        self.gain = GainAugment()

    def forward(self, mel):
        mel = self.gain(mel)
        if self.spec_augment is not None:
            mel = self.spec_augment(mel)
        return mel
