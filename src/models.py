"""
Sound Event Detection (SED) model for BirdCLEF 2026.

Architecture: Backbone → GEM Frequency Pooling → Attention Temporal Pooling
──────────────────────────────────────────────────────────────────────────────

Why SED over clip classification?
  BirdCLEF evaluates on 5-second windows from continuous soundscapes where
  multiple species overlap temporally. A pure clip classifier collapses "species
  present somewhere in the window" into one pooled representation, losing temporal
  evidence. SED preserves frame-level predictions, enabling:
  - Better overlap handling (species active at different times)
  - Temporal evidence for post-processing (peak counting, duration filtering)
  - More informative pseudo-labels (framewise soft targets)

Mathematical components:
────────────────────────
1. GEM (Generalized Mean) Pooling over frequency axis:
   GEM(x; p) = (1/F · Σ_f x_f^p)^{1/p}

   - p = 1 → average pooling (all freq bins equal)
   - p → ∞ → max pooling (only peak frequency)
   - Learnable p adapts to the task: birdsong is often narrowband (favoring
     higher p), while insects/frogs can be broadband (favoring lower p).
   Reference: Radenović et al., "Fine-tuning CNN Image Retrieval with No Human
   Annotation", TPAMI 2019.

2. Attention-weighted temporal pooling:
   Given frame features h_t ∈ ℝ^d for t = 1,...,T:

   a_t = W_a · h_t + b_a          (attention logits, ℝ^C)
   z_t = W_c · h_t + b_c          (class logits, ℝ^C)
   α_t = softmax_t(tanh(a_t))     (attention weights, normalized over time)
   ŷ = Σ_t α_t ⊙ z_t             (clip-level prediction)

   The tanh bounds attention logits to [-1, 1], preventing gradient explosion
   through the softmax. Class-specific attention means different species can
   attend to different time frames — essential for polyphonic soundscapes.

   Reference: Yu et al., "Multi-scale Attention for Audio Tagging", DCASE 2018.

3. Output heads:
   - clip_logits: (B, C) — used for loss computation and final prediction
   - frame_logits: (B, T, C) — used for SED output and temporal post-processing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GEMPool2d(nn.Module):
    """
    Generalized Mean Pooling over a spatial dimension.

    GEM(x; p) = (1/|Ω| · Σ_{i∈Ω} x_i^p)^{1/p}

    Pools over dim (default: frequency axis, dim=2) while preserving other dims.
    When applied after the backbone, reduces (B, C, F', T') → (B, C, T').
    """
    def __init__(self, p_init=3.0, trainable=True, eps=1e-6, dim=2):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)),
                              requires_grad=trainable)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        # Clamp for numerical stability: avoid 0^p and overflow
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=self.dim)
        return x.pow(1.0 / p)


class AttentionHead(nn.Module):
    """
    Class-specific attention pooling over the temporal axis.

    For each class c, learns independent attention weights over time:
      α_{t,c} = softmax_t(tanh(a_{t,c}))
      clip_logit_c = Σ_t α_{t,c} · z_{t,c}

    This is more expressive than global attention (single weight per frame)
    because different species vocalize at different times within a window.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.attention = nn.Linear(in_features, num_classes)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) frame-level features
        Returns:
            clip_logits: (B, C) attention-weighted temporal aggregation
            frame_logits: (B, T, C) per-frame class logits
        """
        att = self.attention(x)             # (B, T, C)
        logit = self.classifier(x)          # (B, T, C)

        # tanh gate → softmax over time
        att_weights = torch.softmax(torch.tanh(att), dim=1)  # (B, T, C)

        # Weighted aggregation
        clip_logits = (att_weights * logit).sum(dim=1)  # (B, C)

        return clip_logits, logit  # clip-level, frame-level


class SEDModel(nn.Module):
    """
    Full SED pipeline: Backbone → GEM → Attention → predictions.

    Input:  (B, 3, n_mels, T)  mel spectrogram (3-channel for pretrained)
    Output: clip_logits (B, C), frame_logits (B, T', C)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Backbone: pretrained CNN feature extractor
        # num_classes=0, global_pool='' → return raw feature maps
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=True,
            in_chans=config.in_chans,
            num_classes=0,
            global_pool="",
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        # Determine feature dimension by forward pass
        with torch.no_grad():
            dummy = torch.randn(1, config.in_chans, config.n_mels, 313)
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[1]  # C
            self._feat_h = feat.shape[2]   # F' (frequency)
            self._feat_w = feat.shape[3]   # T' (time)

        # GEM pooling over frequency axis
        self.gem = GEMPool2d(p_init=config.gem_p_init,
                             trainable=config.gem_p_trainable, dim=2)

        # Dropout before head
        self.dropout = nn.Dropout(config.drop_rate)

        # Attention-based SED head
        self.head = AttentionHead(self.feat_dim, config.num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, 3, n_mels, T) mel spectrogram
        Returns:
            clip_logits: (B, num_classes)
            frame_logits: (B, T', num_classes)
        """
        # Backbone: (B, 3, H, W) → (B, C, F', T')
        features = self.backbone(x)

        # GEM pool frequency: (B, C, F', T') → (B, C, T')
        features = self.gem(features)

        # Transpose: (B, C, T') → (B, T', C) for attention head
        features = features.permute(0, 2, 1)

        features = self.dropout(features)

        # Attention head: → clip (B, num_classes), frame (B, T', num_classes)
        clip_logits, frame_logits = self.head(features)

        return clip_logits, frame_logits

    def get_param_groups(self, config):
        """
        Differential learning rates:
        - Backbone: lr × backbone_lr_mult (preserve pretrained features)
        - Head + GEM: lr (train from scratch)

        Scientific basis: Kornblith et al., "Do Better ImageNet Models Transfer
        Better?" (CVPR 2019) showed that aggressive fine-tuning of early layers
        can destroy transferable features. Differential LR is the standard fix.
        """
        backbone_params = list(self.backbone.parameters())
        head_params = (list(self.gem.parameters()) +
                       list(self.head.parameters()) +
                       list(self.dropout.parameters()))

        return [
            {"params": backbone_params, "lr": config.lr * config.backbone_lr_mult},
            {"params": head_params, "lr": config.lr},
        ]
