"""
Multi-window soundscape dataset for Path 2 training.

Each item is one 60-second soundscape recording, returned as W=12 contiguous
5-second windows along with per-window soft pseudo-labels from the V137 teacher
ensemble. This is the data shape required for cross-window joint training.
"""
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class MultiWindowSoundscapeDataset(Dataset):
    """
    Returns one full soundscape file per item.

    Output:
      mel:   (W, 3, n_mels, T_mel)        per-window mel spectrograms
      label: (W, num_classes)              per-window soft pseudo-labels
      hard:  (W, num_classes)              per-window hard pseudo-labels (binarized)
      mask:  (W,)                          1.0 if window has any high-confidence species

    The dataset reads R4-style raw_predictions.csv (one row per labeled window).
    Files with fewer than W windows are dropped to keep batch shape uniform.
    """
    def __init__(self, raw_predictions_csv, soundscape_dir, label_cols, config,
                 augmentations=None, n_windows=12, hard_threshold_k=1.0,
                 hard_threshold_floor=0.3, hard_threshold_ceil=0.9,
                 hard_label_csv_paths=None, window_mask_prob=0.0,
                 window_mask_max=2):
        self.df = pd.read_csv(raw_predictions_csv)
        self.soundscape_dir = Path(soundscape_dir)
        self.label_cols = list(label_cols)
        self.label_to_idx = {l: i for i, l in enumerate(self.label_cols)}
        self.config = config
        self.augmentations = augmentations
        self.n_windows = n_windows
        self.target_samples = config.target_samples
        # Window masking: zero a random subset of mels in __getitem__. Forces
        # the cross-window attention to fill in masked windows from the rest,
        # which only works if joint reasoning is engaged.
        self.window_mask_prob = float(window_mask_prob)
        self.window_mask_max = int(window_mask_max)

        # Adaptive per-class thresholds (matches PseudoLabelDataset logic)
        self.thresholds = np.full(len(self.label_cols), 0.9, dtype=np.float32)
        for i, col in enumerate(self.label_cols):
            if col in self.df.columns:
                vals = self.df[col].values
                mu, sigma = float(vals.mean()), float(vals.std())
                self.thresholds[i] = float(np.clip(
                    mu + hard_threshold_k * sigma,
                    hard_threshold_floor, hard_threshold_ceil,
                ))

        # Optional: union with labeled soundscape ground truth for the 60 known files
        self.hard_overrides = {}  # (filename, end_time) -> set(label_idx)
        if hard_label_csv_paths:
            for p in hard_label_csv_paths:
                p = Path(p)
                if not p.exists():
                    continue
                lf = pd.read_csv(p).drop_duplicates()
                lf["primary_label"] = lf["primary_label"].astype(str)
                lf["end_sec"] = pd.to_timedelta(lf["end"]).dt.total_seconds().astype(int)
                for _, row in lf.iterrows():
                    key = (row["filename"], int(row["end_sec"]))
                    self.hard_overrides.setdefault(key, set())
                    for lbl in str(row["primary_label"]).split(";"):
                        lbl = lbl.strip()
                        if lbl in self.label_to_idx:
                            self.hard_overrides[key].add(self.label_to_idx[lbl])

        # Group rows by filename, sort by end_time → keep only files with W windows
        self.df["end_time"] = self.df["end_time"].astype(float)
        groups = self.df.sort_values(["file", "end_time"]).groupby("file")

        self.files = []  # list of (filename, soft_W_C, hard_W_C, ends_W)
        for filename, g in groups:
            if len(g) < n_windows:
                continue
            g = g.head(n_windows)
            soft = g[self.label_cols].values.astype(np.float32)  # (W, C)
            hard = (soft >= self.thresholds[None, :]).astype(np.float32)
            ends = g["end_time"].values.astype(np.float32)
            # Apply overrides
            for w in range(n_windows):
                key = (filename, int(ends[w]))
                if key in self.hard_overrides:
                    for ci in self.hard_overrides[key]:
                        hard[w, ci] = 1.0
            self.files.append((filename, soft, hard, ends))

        print(f"MultiWindowSoundscape: {len(self.files)} files × {n_windows} windows "
              f"(threshold range {self.thresholds.min():.2f}–{self.thresholds.max():.2f})")

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate, n_fft=config.n_fft,
            hop_length=config.hop_length, n_mels=config.n_mels,
            f_min=config.fmin, f_max=config.fmax,
            power=2.0, norm="slaney", mel_scale="htk",
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename, soft, hard, ends = self.files[idx]
        path = self.soundscape_dir / filename
        sr = self.config.sample_rate

        wav, fsr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if fsr != sr:
            wav = torchaudio.functional.resample(wav, fsr, sr)

        target_total = int(self.n_windows * self.target_samples)
        if wav.shape[-1] < target_total:
            wav = torch.nn.functional.pad(wav, (0, target_total - wav.shape[-1]))
        wav = wav[..., :target_total]

        windows = wav.reshape(1, self.n_windows, self.target_samples).permute(1, 0, 2)
        # windows: (W, 1, target_samples)

        mels = []
        for w in range(self.n_windows):
            mel = self.mel_transform(windows[w])
            mel = self.db_transform(mel)
            m_min, m_max = mel.min(), mel.max()
            mel = (mel - m_min) / (m_max - m_min + 1e-8)
            mel = mel.expand(3, -1, -1)
            if self.augmentations is not None:
                mel = self.augmentations(mel)
            mels.append(mel)
        mels = torch.stack(mels, dim=0)  # (W, 3, n_mels, T_mel)

        # Optional window-mask augmentation: zero out a few windows so the
        # model must rely on cross-window attention to predict their labels.
        if self.window_mask_prob > 0 and self.window_mask_max > 0:
            if torch.rand(1).item() < self.window_mask_prob:
                k = torch.randint(1, self.window_mask_max + 1, (1,)).item()
                idx = torch.randperm(self.n_windows)[:k]
                mels[idx] = 0.0

        soft_t = torch.from_numpy(soft)
        hard_t = torch.from_numpy(hard)
        # Window mask: at least one species above threshold or any override
        mask = (hard_t.sum(dim=-1) > 0).float()

        return mels, soft_t, hard_t, mask
