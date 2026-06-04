"""
5-stage post-processing stack for BirdCLEF 2026.

Each stage is mathematically motivated and independently ablatable.

Pipeline:
  Raw probabilities p(t, c)
    ↓
  Stage 1: Per-taxon temperature scaling
    → Calibrates confidence per taxonomic group
    ↓
  Stage 2: File-level confidence scaling
    → Propagates file-wide species presence to all windows
    ↓
  Stage 3: Rank-aware scaling
    → Suppresses uncertain files, boosts confident ones
    ↓
  Stage 4: Delta shift smoothing
    → Temporal averaging with neighbors
    ↓
  Stage 5: Per-class threshold sharpening
    → OOF-optimized nonlinear rescaling
    ↓
  Final probabilities

References:
  - Stage 2: BirdCLEF 2024 winning solutions (file-max leakage)
  - Stage 3: BirdCLEF 2025 3rd-place (rank-aware scaling)
  - Stage 4: BirdCLEF 2025 1st-place (delta shift smoothing)
  - Stage 5: Platt scaling variant (post-hoc calibration)
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def apply_postprocessing(
    probs,
    taxonomy_map=None,
    oof_thresholds=None,
    # Stage 1 params
    taxon_temperatures=None,
    # Stage 2 params
    file_max_alpha=0.05,
    # Stage 3 params
    rank_power=0.4,
    rank_top_k=5,
    # Stage 4 params
    delta_alpha=0.20,
    # Stage 5 params
    threshold_sharpening=True,
):
    """
    Apply full post-processing pipeline.

    Args:
        probs: (N_files, T=12, C=234) raw probabilities [0, 1]
        taxonomy_map: dict mapping class_idx → taxon_name (for stage 1)
        oof_thresholds: (C,) per-class thresholds from OOF (for stage 5)
        other params: see stage descriptions

    Returns:
        probs: (N_files, T, C) post-processed probabilities
    """
    probs = probs.copy()

    # Stage 1: Per-taxon temperature scaling
    if taxon_temperatures is not None and taxonomy_map is not None:
        probs = stage1_taxon_temperature(probs, taxonomy_map, taxon_temperatures)

    # Stage 2: File-level confidence scaling
    if file_max_alpha > 0:
        probs = stage2_file_max_prior(probs, alpha=file_max_alpha)

    # Stage 3: Rank-aware scaling
    if rank_power > 0:
        probs = stage3_rank_aware(probs, power=rank_power, top_k=rank_top_k)

    # Stage 4: Delta shift smoothing
    if delta_alpha > 0:
        probs = stage4_delta_smoothing(probs, alpha=delta_alpha)

    # Stage 5: Per-class threshold sharpening
    if threshold_sharpening and oof_thresholds is not None:
        probs = stage5_threshold_sharpening(probs, oof_thresholds)

    return probs


# ═══════════════════════════════════════════════════════════════════════════════

def stage1_taxon_temperature(probs, taxonomy_map, taxon_temperatures):
    """
    Scale probabilities by taxon-specific temperature.

    Motivation: Different taxa have different confidence characteristics.
    Birds (event-like, short bursts) tend to have sharper predictions.
    Frogs/insects (texture-like, continuous) tend to be more diffuse.

    Math: p_scaled = sigmoid(logit(p) / T)
    where T > 1 cools (less confident) and T < 1 sharpens (more confident).
    """
    # Default temperatures (from ProtoSSM v5 literature)
    default_temps = {
        'Aves': 1.10,       # slightly cooled — birds overconfident
        'Amphibia': 0.95,   # slightly sharpened — frogs need more confidence
        'Insecta': 0.95,    # same for insects
        'Mammalia': 1.0,    # neutral
        'Reptilia': 1.0,    # neutral
    }
    temps = taxon_temperatures or default_temps

    eps = 1e-7
    logits = np.log(np.clip(probs, eps, 1 - eps) / (1 - np.clip(probs, eps, 1 - eps)))

    for c_idx, taxon in taxonomy_map.items():
        T = temps.get(taxon, 1.0)
        if T != 1.0:
            logits[..., c_idx] = logits[..., c_idx] / T

    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs


def stage2_file_max_prior(probs, alpha=0.05):
    """
    File-level max prior: species persistence assumption.

    If a species appears confidently in ANY window of a file,
    it likely persists (faintly) in nearby windows too.

    Math: p̃(t, c) = p(t, c) + α × max_t' p(t', c)

    This is a conservative prior: α=0.05 adds 5% of the file-wide
    maximum confidence. Helps catch faint/unannotated calls.
    """
    # probs: (N_files, T, C)
    file_max = probs.max(axis=1, keepdims=True)  # (N_files, 1, C)
    probs = probs + alpha * file_max
    return probs


def stage3_rank_aware(probs, power=0.4, top_k=5):
    """
    Rank-aware scaling: suppress uncertain files, boost confident ones.

    Motivation: If the top-K species in a file have low confidence,
    the file is likely silent/noisy. Scale all predictions down.
    If top-K are high confidence, scale up.

    Math: scale_c = max_t(p(t,c))^γ   (per-class, per-file)
          p_scaled(t, c) = p(t, c) × scale_c
    """
    file_max = probs.max(axis=1)  # (N_files, C)
    scale = np.power(np.clip(file_max, 1e-7, None), power)  # (N_files, C)
    probs = probs * scale[:, np.newaxis, :]  # broadcast over T
    return probs


def stage4_delta_smoothing(probs, alpha=0.20):
    """
    Delta shift smoothing: temporal averaging with neighbors.

    Motivation: Bird calls often span multiple 5s windows.
    Smoothing reduces noise from window boundary artifacts.

    Math: p̃(t) = (1-α)·p(t) + α/2·(p(t-1) + p(t+1))

    Adaptive variant: confident windows get less smoothing.
    """
    N, T, C = probs.shape
    smoothed = probs.copy()

    for t in range(T):
        prev_t = max(0, t - 1)
        next_t = min(T - 1, t + 1)

        # Adaptive: scale alpha by (1 - confidence) so confident windows smooth less
        # confidence = max prediction at this window across all species
        confidence = probs[:, t, :].max(axis=-1, keepdims=True)  # (N, 1)
        alpha_eff = alpha * (1.0 - confidence)  # lower for confident windows

        neighbor_avg = 0.5 * (probs[:, prev_t, :] + probs[:, next_t, :])
        smoothed[:, t, :] = (1 - alpha_eff) * probs[:, t, :] + alpha_eff * neighbor_avg

    return smoothed


def stage5_threshold_sharpening(probs, thresholds):
    """
    Per-class threshold sharpening using OOF-optimized thresholds.

    Motivation: AUC is rank-based, so we want to separate positives
    from negatives as cleanly as possible. Sharpening around the
    optimal threshold amplifies the separation.

    Math: For class c with threshold θ_c:
      if p > θ: p̃ = 0.5 + 0.5·(p - θ)/(1 - θ)    → mapped to [0.5, 1.0]
      if p ≤ θ: p̃ = 0.5·p/θ                        → mapped to [0.0, 0.5]

    This is a piecewise linear rescaling that preserves ranking
    within each partition but pushes scores away from the threshold.
    """
    eps = 1e-7
    C = probs.shape[-1]
    result = probs.copy()

    for c in range(C):
        theta = thresholds[c]
        if theta <= 0 or theta >= 1:
            continue

        above = probs[..., c] > theta
        below = ~above

        result[..., c] = np.where(
            above,
            0.5 + 0.5 * (probs[..., c] - theta) / (1 - theta + eps),
            0.5 * probs[..., c] / (theta + eps),
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: compute optimal thresholds from OOF predictions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_oof_thresholds(y_true, y_pred, n_thresholds=50):
    """
    Find optimal per-class thresholds that maximize per-class AUC separation.

    Uses the F1-optimal threshold as a proxy for the best separation point.
    """
    from sklearn.metrics import f1_score
    C = y_true.shape[1]
    thresholds = np.full(C, 0.5)

    for c in range(C):
        if y_true[:, c].sum() == 0:
            continue

        best_f1 = 0
        for t in np.linspace(0.01, 0.99, n_thresholds):
            f1 = f1_score(y_true[:, c], (y_pred[:, c] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                thresholds[c] = t

    return thresholds
