"""
Utilities: metrics, seeding, logging.

The core metric is macro-averaged ROC-AUC, matching the BirdCLEF+ 2026
evaluation. This metric measures ranking quality per class, averaged across
classes. Key properties:

1. Threshold-free: AUC measures the probability that a randomly chosen positive
   is ranked higher than a randomly chosen negative. No operating threshold needed.

2. Macro-averaged: Each class gets equal weight regardless of sample count.
   This means a rare species with 2 recordings matters as much as a common
   species with 499. This is why long-tail performance is critical.

3. Classes with no positives are skipped (competition rule). This avoids
   undefined AUC for species absent from the test set.
"""
import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    """Deterministic training for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Keep False for speed
    torch.backends.cudnn.benchmark = True        # Optimize for fixed input size


def macro_auc(y_true, y_pred):
    """
    Competition metric: macro-averaged ROC-AUC, skipping classes with no positives.

    Args:
        y_true: (N, C) binary ground truth
        y_pred: (N, C) predicted probabilities
    Returns:
        mean AUC across classes with at least one positive
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0:
            continue  # Skip classes with no positives
        if col.sum() == col.shape[0]:
            continue  # Skip classes with all positives (undefined AUC)
        try:
            auc = roc_auc_score(col, y_pred[:, c])
            aucs.append(auc)
        except ValueError:
            continue
    return np.mean(aucs) if aucs else 0.0


class AverageMeter:
    """Running average tracker for training metrics."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
