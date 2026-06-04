"""
Loss functions for multi-label bioacoustic classification.

The choice of loss function is critical for BirdCLEF because:
1. Extreme class imbalance: 234 classes, >99% of labels are negative per sample
2. Long-tailed distribution: species counts range from 1 to 499
3. Metric is macro AUC: ranking quality matters, not threshold accuracy

Three loss functions are provided, in order of expected effectiveness:

═══════════════════════════════════════════════════════════════════════════════
1. Asymmetric Loss (ASL) — RECOMMENDED for BirdCLEF
═══════════════════════════════════════════════════════════════════════════════

  L_ASL = -(1/C) Σ_c [ y_c · (1-p_c)^γ+ · log(p_c)
                      + (1-y_c) · p̂_c^γ- · log(1-p̂_c) ]

  where p̂_c = max(p_c - m, 0) is the shifted probability.

  Key innovations over standard BCE:
  a) Probability shifting (margin m): Sets gradient to zero for negative
     predictions below m. This creates a "dead zone" where the model isn't
     penalized for slightly positive predictions on negatives — reducing the
     overwhelming gradient signal from easy negatives.

  b) Asymmetric focusing (γ+ < γ-): The standard focal loss uses one γ for
     both. ASL decouples them: γ+=0 means no focusing on positives (every
     positive matters equally), γ-=4 strongly down-weights easy negatives.

  Why this matters for BirdCLEF: with 234 classes and typically 1-3 positive
  per sample, there are ~231 negative classes per sample. Without ASL, the
  gradient is dominated by these negatives. ASL shifts the gradient budget
  toward the rare positive signals.

  Reference: Ridnik et al., "Asymmetric Loss For Multi-Label Classification",
  ICCV 2021.

═══════════════════════════════════════════════════════════════════════════════
2. Focal Loss
═══════════════════════════════════════════════════════════════════════════════

  L_focal = -(1/C) Σ_c [ y_c · (1-p_c)^γ · log(p_c)
                        + (1-y_c) · p_c^γ · log(1-p_c) ]

  Symmetric: same γ for positives and negatives. Less effective than ASL
  for the extreme imbalance in BirdCLEF, but a reasonable baseline.

  Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

═══════════════════════════════════════════════════════════════════════════════
3. Knowledge Distillation Loss
═══════════════════════════════════════════════════════════════════════════════

  L_KD = α · L_hard(z_S, y) + (1-α) · τ² · D_KL(σ(z_T/τ) ‖ σ(z_S/τ))

  The temperature τ softens the probability distributions, exposing inter-class
  relationships learned by the teacher. The τ² factor compensates for the
  reduced gradient magnitude when logits are scaled by 1/τ.

  For multi-label (independent sigmoids), KL divergence decomposes per-class:
  D_KL = Σ_c [ q_c · log(q_c/p_c) + (1-q_c) · log((1-q_c)/(1-p_c)) ]
  where q_c = σ(z_T,c/τ) and p_c = σ(z_S,c/τ).

  Reference: Hinton et al., "Distilling the Knowledge in a Neural Network",
  NIPS 2014 Workshop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    ASL for multi-label classification.

    Args:
        gamma_pos: Focusing parameter for positive samples (typically 0)
        gamma_neg: Focusing parameter for negative samples (typically 2-4)
        clip_margin: Probability shift for negatives (typically 0.05)
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip_margin=0.05, eps=1e-7):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip_margin = clip_margin
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw logits (before sigmoid)
            targets: (B, C) multi-hot labels in {0, 1}
        """
        probs = torch.sigmoid(logits)

        # Separate positive and negative
        pos_probs = probs
        neg_probs = (probs - self.clip_margin).clamp(min=self.eps)

        # Asymmetric focusing
        pos_loss = targets * (1 - pos_probs).pow(self.gamma_pos) * torch.log(pos_probs.clamp(min=self.eps))
        neg_loss = (1 - targets) * neg_probs.pow(self.gamma_neg) * torch.log((1 - neg_probs).clamp(min=self.eps))

        loss = -(pos_loss + neg_loss)
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for multi-label classification.

    L = -α_t · (1-p_t)^γ · log(p_t)
    where p_t = p if y=1, else 1-p.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        return (focal_weight * ce).mean()


class DistillationLoss(nn.Module):
    """
    Combined hard-label + soft-label distillation loss.

    For multi-label (sigmoid), we use binary KL divergence per class:
      D_KL(q ‖ p) = q·log(q/p) + (1-q)·log((1-q)/(1-p))
    """
    def __init__(self, hard_loss_fn, alpha=0.5, temperature=3.0):
        super().__init__()
        self.hard_loss_fn = hard_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, targets, teacher_logits):
        # Hard loss: student vs ground truth
        hard_loss = self.hard_loss_fn(student_logits, targets)

        # Soft loss: student vs teacher (temperature-scaled)
        tau = self.temperature
        teacher_probs = torch.sigmoid(teacher_logits / tau)
        student_log_probs = F.logsigmoid(student_logits / tau)
        student_log_1m = F.logsigmoid(-student_logits / tau)  # log(1 - σ(z/τ))

        # Binary KL divergence per class
        soft_loss = (
            teacher_probs * (torch.log(teacher_probs.clamp(min=1e-7)) - student_log_probs)
            + (1 - teacher_probs) * (torch.log((1 - teacher_probs).clamp(min=1e-7)) - student_log_1m)
        )
        soft_loss = tau * tau * soft_loss.mean()  # τ² scaling

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


class SoftAUCLoss(nn.Module):
    """
    Differentiable approximation of macro AUC loss.

    For each class, AUC = P(score(pos) > score(neg)). We approximate this with:
      softAUC_c = (1/|P_c||N_c|) Σ_{i∈P_c} Σ_{j∈N_c} σ((s_i - s_j) / τ)

    Loss = 1 - mean(softAUC_c) across classes with both positives and negatives.

    Reference: Used by 1st place BirdCLEF 2025 for direct metric optimization.
    """
    def __init__(self, temperature=1.0, eps=1e-7):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw logits (before sigmoid)
            targets: (B, C) multi-hot labels in {0, 1}
        """
        B, C = logits.shape
        # Vectorized: process all classes with enough pos/neg in one pass
        pos_mask = targets > 0.5  # (B, C)
        neg_mask = targets < 0.5  # (B, C)
        n_pos = pos_mask.sum(dim=0)  # (C,)
        n_neg = neg_mask.sum(dim=0)  # (C,)
        valid = (n_pos > 0) & (n_neg > 0)  # (C,)

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        auc_sum = torch.tensor(0.0, device=logits.device)
        # Process valid classes — still need loop for variable-size pos/neg sets
        # but skip invalid classes early
        valid_idx = valid.nonzero(as_tuple=True)[0]
        for c in valid_idx:
            pos_logits = logits[pos_mask[:, c], c]
            neg_logits = logits[neg_mask[:, c], c]
            diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
            auc_sum = auc_sum + torch.sigmoid(diff / self.temperature).mean()

        return 1.0 - auc_sum / len(valid_idx)


class CombinedASLAUCLoss(nn.Module):
    """
    Combined ASL + SoftAUC loss for joint optimization.

    ASL handles the per-sample gradient well, while SoftAUC directly optimizes
    the competition metric. Combining them gives both local gradient quality
    and global ranking optimization.
    """
    def __init__(self, asl_weight=0.7, auc_weight=0.3,
                 gamma_pos=0.0, gamma_neg=4.0, clip_margin=0.05,
                 temperature=1.0):
        super().__init__()
        self.asl = AsymmetricLoss(gamma_pos, gamma_neg, clip_margin)
        self.auc = SoftAUCLoss(temperature)
        self.asl_weight = asl_weight
        self.auc_weight = auc_weight

    def forward(self, logits, targets):
        return self.asl_weight * self.asl(logits, targets) + \
               self.auc_weight * self.auc(logits, targets)


def build_loss(config):
    """Factory function for loss construction."""
    if config.loss_type == "combined_asl_auc":
        return CombinedASLAUCLoss(
            asl_weight=0.7, auc_weight=0.3,
            gamma_pos=config.gamma_pos, gamma_neg=config.gamma_neg,
            clip_margin=config.clip_margin,
        )

    if config.loss_type == "asl":
        hard_loss = AsymmetricLoss(
            gamma_pos=config.gamma_pos,
            gamma_neg=config.gamma_neg,
            clip_margin=config.clip_margin,
        )
    elif config.loss_type == "focal":
        hard_loss = FocalLoss()
    else:
        hard_loss = nn.BCEWithLogitsLoss()

    if config.teacher_weight > 0 and config.teacher_checkpoint:
        return DistillationLoss(
            hard_loss_fn=hard_loss,
            alpha=1.0 - config.teacher_weight,
            temperature=config.temperature,
        )
    return hard_loss
