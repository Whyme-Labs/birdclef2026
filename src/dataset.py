"""
BirdCLEF 2026 Dataset.

Handles audio loading, mel-spectrogram extraction, and label encoding.

Mathematical foundations:
─────────────────────────
Mel spectrogram:
  S(m, t) = log(1 + M · |STFT(x)|²)
  where M ∈ ℝ^{n_mels × (n_fft/2+1)} is the mel filterbank matrix.
  Mel scale (HTK): m = 2595 · log₁₀(1 + f/700)

For 5s audio at 32kHz with n_fft=2048, hop=512:
  - Input samples: 160,000
  - STFT frames (center-padded): ⌊160000/512⌋ + 1 = 313
  - Output shape: (n_mels, 313) = (224, 313)

We repeat to 3 channels for ImageNet-pretrained backbones. This is not a hack —
it's well-supported by transfer learning theory: the first-layer conv filters
still extract useful edge/texture features from spectrograms.
"""
import ast
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold


class BirdCLEFDataset(Dataset):
    def __init__(self, df, audio_dir, label_cols, config,
                 is_train=True, augmentations=None, extra_audio_dirs=None):
        """
        extra_audio_dirs: optional dict mapping `source` tag → Path. If `df` has
        a `source` column, rows are resolved to extra_audio_dirs[source]/filename;
        otherwise they fall back to `audio_dir / filename`. This enables a single
        dataset to serve both 2026 and external 2025 audio without breaking the
        existing code path (no `source` column → unchanged behavior).
        """
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.extra_audio_dirs = (
            {k: Path(v) for k, v in extra_audio_dirs.items()}
            if extra_audio_dirs else None
        )
        self.has_source = "source" in self.df.columns
        self.label_cols = label_cols
        self.label_to_idx = {l: i for i, l in enumerate(label_cols)}
        self.config = config
        self.is_train = is_train
        self.augmentations = augmentations
        self.target_samples = config.target_samples

        # Mel spectrogram: STFT → mel filterbank → log scale
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.fmin,
            f_max=config.fmax,
            power=2.0,
            norm="slaney",
            mel_scale="htk",
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = self._resolve_path(row)

        # Load and preprocess audio
        waveform = self._load_audio(audio_path)
        waveform = self._crop_or_pad(waveform)

        # Waveform augmentations (before mel) — training only
        if self.is_train:
            waveform = self._waveform_augment(waveform)

        # Waveform → mel spectrogram → dB → normalize
        mel = self.mel_transform(waveform)      # (1, n_mels, T)
        mel = self.db_transform(mel)             # (1, n_mels, T) in dB

        # Per-sample min-max normalization to [0, 1]
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)

        # Repeat to 3 channels for pretrained backbone
        mel = mel.expand(3, -1, -1)              # (3, n_mels, T)

        # Build multi-hot label
        label = self._encode_label(row)

        # Apply augmentations (SpecAugment, etc.) — mixup handled at batch level
        if self.is_train and self.augmentations is not None:
            mel = self.augmentations(mel)

        return mel, label

    def _resolve_path(self, row):
        """Resolve the audio path for a row.

        If the dataframe has a `source` column and extra_audio_dirs is
        configured, look up the audio base by source; else fall back to
        the default self.audio_dir. This keeps backward compatibility
        with train.csv (no `source` column).
        """
        if self.has_source and self.extra_audio_dirs is not None:
            # `source` may be int (e.g. 2025/2026) or str — normalize to str
            # since extra_audio_dirs is keyed by string.
            src = str(row["source"])
            base = self.extra_audio_dirs.get(src, self.audio_dir)
            return Path(base) / row["filename"]
        return self.audio_dir / row["filename"]

    def _load_audio(self, path):
        """Load audio, convert to mono, resample to target SR."""
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.config.sample_rate
            )
        return waveform

    def _waveform_augment(self, waveform):
        """Waveform-level augmentations before mel conversion."""
        # Gaussian noise injection (simulates recorder/ambient noise)
        if torch.rand(1).item() < 0.5:
            snr_db = torch.empty(1).uniform_(15, 35).item()
            sig_power = waveform.pow(2).mean()
            noise_power = sig_power / (10 ** (snr_db / 10))
            waveform = waveform + torch.randn_like(waveform) * noise_power.sqrt()
        # Pink noise (more realistic than white for outdoor recordings)
        if torch.rand(1).item() < 0.3:
            snr_db = torch.empty(1).uniform_(20, 40).item()
            n = waveform.shape[-1]
            freqs = torch.fft.rfftfreq(n) + 1e-6
            pink_spectrum = torch.randn(waveform.shape[0], freqs.shape[0]) / freqs.sqrt()
            pink_noise = torch.fft.irfft(pink_spectrum, n=n)
            sig_power = waveform.pow(2).mean()
            noise_power = sig_power / (10 ** (snr_db / 10))
            pink_noise = pink_noise * (noise_power.sqrt() / (pink_noise.pow(2).mean().sqrt() + 1e-8))
            waveform = waveform + pink_noise
        return waveform

    def _crop_or_pad(self, waveform):
        """
        Training: random crop from the recording.
        Validation: center crop.
        Short clips: tile-pad to target length.
        """
        n = waveform.shape[-1]
        target = self.target_samples
        if n > target:
            if self.is_train:
                start = torch.randint(0, n - target, (1,)).item()
            else:
                start = (n - target) // 2
            waveform = waveform[..., start:start + target]
        elif n < target:
            reps = (target // n) + 1
            waveform = waveform.repeat(1, reps)[..., :target]
        return waveform

    def _encode_label(self, row):
        """Build multi-hot label vector over all 234 target species."""
        label = torch.zeros(len(self.label_cols), dtype=torch.float32)
        primary = str(row["primary_label"])
        if primary in self.label_to_idx:
            label[self.label_to_idx[primary]] = 1.0
        # Secondary labels
        sec = row.get("secondary_labels", "[]")
        if isinstance(sec, str) and sec not in ("[]", ""):
            try:
                for s in ast.literal_eval(sec):
                    s = str(s)
                    if s in self.label_to_idx:
                        label[self.label_to_idx[s]] = 1.0
            except (ValueError, SyntaxError):
                pass
        return label


class SoundscapeDataset(Dataset):
    """
    Dataset for labeled soundscape windows.

    Each row in the labels CSV specifies a file, start/end time, and
    semicolon-separated species labels. This covers the 28 species
    with zero training audio recordings.
    """
    def __init__(self, labels_csv, soundscape_dir, label_cols, config,
                 is_train=True, augmentations=None):
        self.df = pd.read_csv(labels_csv)
        self.soundscape_dir = Path(soundscape_dir)
        self.label_cols = label_cols
        self.label_to_idx = {l: i for i, l in enumerate(label_cols)}
        self.config = config
        self.is_train = is_train
        self.augmentations = augmentations
        self.target_samples = config.target_samples

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate, n_fft=config.n_fft,
            hop_length=config.hop_length, n_mels=config.n_mels,
            f_min=config.fmin, f_max=config.fmax,
            power=2.0, norm="slaney", mel_scale="htk",
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # Cache loaded audio to avoid re-reading same soundscape files
        self._audio_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        audio_path = self.soundscape_dir / filename

        # Parse time range
        start_sec = self._parse_time(row["start"])
        end_sec = self._parse_time(row["end"])

        # Load and extract chunk
        waveform = self._load_chunk(audio_path, start_sec, end_sec)

        # Mel spectrogram
        mel = self.mel_transform(waveform)
        mel = self.db_transform(mel)
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)
        mel = mel.expand(3, -1, -1)

        # Multi-label encoding from semicolon-separated species
        label = torch.zeros(len(self.label_cols), dtype=torch.float32)
        species = str(row["primary_label"]).split(";")
        for sp in species:
            sp = sp.strip()
            if sp in self.label_to_idx:
                label[self.label_to_idx[sp]] = 1.0

        if self.is_train and self.augmentations is not None:
            mel = self.augmentations(mel)

        return mel, label

    @staticmethod
    def _parse_time(t):
        """Parse HH:MM:SS to seconds."""
        parts = str(t).split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    def _load_chunk(self, path, start_sec, end_sec):
        sr = self.config.sample_rate
        start_sample = int(start_sec * sr)

        # Load full file (cached), then slice
        if str(path) not in self._audio_cache:
            wav, fsr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if fsr != sr:
                wav = torchaudio.functional.resample(wav, fsr, sr)
            self._audio_cache[str(path)] = wav
            # Limit cache size
            if len(self._audio_cache) > 100:
                oldest = next(iter(self._audio_cache))
                del self._audio_cache[oldest]

        waveform = self._audio_cache[str(path)]
        chunk = waveform[..., start_sample:start_sample + self.target_samples]

        # Pad if needed
        if chunk.shape[-1] < self.target_samples:
            pad = self.target_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        return chunk


class PseudoLabelDataset(Dataset):
    """
    Dataset for pseudo-labeled soundscape windows with HARD-THRESHOLDED labels.

    Critical design decision: use HARD labels (binarized at per-class threshold),
    NOT raw soft teacher predictions.

    Why hard thresholding instead of raw soft labels:
    ─────────────────────────────────────────────────
    The teacher ensemble assigns small non-zero probabilities to absent species
    (e.g., 0.02-0.05). Training on these unfiltered soft labels causes the student
    to learn an elevated baseline prediction for all species, destroying the
    sparsity and calibration needed for AUC-based ranking.

    Mathematically: AUC = P(score_pos > score_neg). If the student learns to
    predict score_neg ≈ 0.05 instead of ≈ 0.001, true positives need much higher
    scores to rank above the inflated negatives. This directly hurts macro AUC.

    Evidence:
    - Lasseck (BirdCLEF 2024) used hard pseudo-labels with per-class thresholds
    - The ICML 2023 SFDA paper warned that uncontrolled adaptation can underperform
      no adaptation — our previous attempt confirmed this (0.853 LB vs 0.862 baseline)
    - Hinton (2015) distillation uses temperature > 1 to control softness, not raw
      teacher outputs which include calibration noise

    The per-class threshold θ_c = clip(μ_c + k·σ_c, θ_min, θ_max) adapts to each
    species' prevalence in the target domain, avoiding a single global threshold
    that would miss rare species.
    """
    def __init__(self, raw_predictions_csv, soundscape_dir, label_cols, config,
                 augmentations=None, label_weight=0.5):
        self.df = pd.read_csv(raw_predictions_csv)
        self.soundscape_dir = Path(soundscape_dir)
        self.label_cols = label_cols
        self.config = config
        self.augmentations = augmentations
        self.label_weight = label_weight
        self.target_samples = config.target_samples

        # Per-class adaptive thresholding: θ_c = clip(μ_c + k·σ_c, θ_min, θ_max)
        # k=1.0 is mildly conservative — keeps high-confidence predictions only
        self.thresholds = {}
        for col in label_cols:
            if col in self.df.columns:
                vals = self.df[col].values
                mu, sigma = vals.mean(), vals.std()
                self.thresholds[col] = np.clip(mu + 1.0 * sigma, 0.3, 0.9)

        # Filter: keep only windows where at least one species exceeds threshold
        has_positive = np.zeros(len(self.df), dtype=bool)
        for col in label_cols:
            if col in self.df.columns and col in self.thresholds:
                has_positive |= (self.df[col].values >= self.thresholds[col])
        self.df = self.df[has_positive].reset_index(drop=True)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate, n_fft=config.n_fft,
            hop_length=config.hop_length, n_mels=config.n_mels,
            f_min=config.fmin, f_max=config.fmax,
            power=2.0, norm="slaney", mel_scale="htk",
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self._audio_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["file"]
        end_time = float(row["end_time"])
        start_sec = end_time - self.config.duration

        # Load audio chunk
        audio_path = self.soundscape_dir / filename
        waveform = self._load_chunk(audio_path, start_sec)

        # Mel spectrogram
        mel = self.mel_transform(waveform)
        mel = self.db_transform(mel)
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)
        mel = mel.expand(3, -1, -1)

        # HARD-THRESHOLDED labels: binarize at per-class adaptive threshold
        # This ensures the student learns clean, sparse predictions
        label = torch.zeros(len(self.label_cols), dtype=torch.float32)
        for i, col in enumerate(self.label_cols):
            if col in self.df.columns and col in self.thresholds:
                if float(row[col]) >= self.thresholds[col]:
                    label[i] = 1.0

        if self.augmentations is not None:
            mel = self.augmentations(mel)

        return mel, label

    def _load_chunk(self, path, start_sec):
        sr = self.config.sample_rate
        start_sample = int(start_sec * sr)

        if str(path) not in self._audio_cache:
            wav, fsr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if fsr != sr:
                wav = torchaudio.functional.resample(wav, fsr, sr)
            self._audio_cache[str(path)] = wav
            if len(self._audio_cache) > 200:
                oldest = next(iter(self._audio_cache))
                del self._audio_cache[oldest]

        waveform = self._audio_cache[str(path)]
        chunk = waveform[..., start_sample:start_sample + self.target_samples]
        if chunk.shape[-1] < self.target_samples:
            pad = self.target_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
        return chunk


def build_splits(train_csv, taxonomy_csv, config):
    """
    Build stratified k-fold splits.

    Returns: (df_train, df_val, label_cols)
    where label_cols is the canonical 234-species list from taxonomy.csv.
    """
    taxonomy = pd.read_csv(taxonomy_csv)
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())

    df = pd.read_csv(train_csv)
    df["primary_label"] = df["primary_label"].astype(str)

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True,
                          random_state=config.seed)
    splits = list(skf.split(df, df["primary_label"]))
    train_idx, val_idx = splits[config.fold]

    return df.iloc[train_idx], df.iloc[val_idx], label_cols


def build_sampler(df, label_cols):
    """
    Inverse-sqrt frequency sampler for class-balanced training.

    Weight for species c: w_c = 1 / sqrt(n_c)
    This is less aggressive than full inverse weighting (1/n_c) but still
    significantly boosts rare species. Empirically superior for long-tail
    recognition (Kang et al., "Decoupling Representation and Classifier", ICLR 2020).
    """
    counts = df["primary_label"].astype(str).value_counts()
    weights = []
    for _, row in df.iterrows():
        pl = str(row["primary_label"])
        n = counts.get(pl, 1)
        weights.append(1.0 / np.sqrt(max(n, 1)))
    weights = torch.tensor(weights, dtype=torch.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_concat_sampler(focal_df, label_cols, soundscape_len=0, pseudo_len=0,
                         power=0.5, soundscape_weight=3.0, pseudo_weight=0.5):
    """
    Multi-source weighted sampler for ConcatDataset [focal + soundscape + pseudo].

    Computes per-sample weights:
    - Focal: 1/n_c^power (class-balanced)
    - Soundscape: mean_class_weight * soundscape_weight (oversampled for domain adaptation)
    - Pseudo: mean_class_weight * pseudo_weight (down-weighted for noise)
    """
    counts = focal_df["primary_label"].astype(str).value_counts()
    # Cap at 500, floor at 10
    capped = {k: np.clip(v, 10, 500) for k, v in counts.items()}

    # Class weights: 1/n^power
    class_w = {}
    for sp in label_cols:
        n = capped.get(sp, 10)
        class_w[sp] = 1.0 / (n ** power) if power > 0 else 1.0

    mean_cw = np.mean(list(class_w.values()))

    # Focal weights
    focal_weights = []
    for _, row in focal_df.iterrows():
        pl = str(row["primary_label"])
        focal_weights.append(class_w.get(pl, mean_cw))

    # Soundscape weights (oversampled for 28 zero-shot species + domain adaptation)
    sc_weights = [mean_cw * soundscape_weight] * soundscape_len

    # Pseudo weights (down-weighted)
    ps_weights = [mean_cw * pseudo_weight] * pseudo_len

    all_w = focal_weights + sc_weights + ps_weights
    all_w = torch.tensor(all_w, dtype=torch.float64)
    return WeightedRandomSampler(all_w, num_samples=len(all_w), replacement=True)
