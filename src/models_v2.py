"""
SED Model V2 — matches the proven public baseline architecture exactly.

Key differences from our v1:
1. AttentionSEDHead has an FC projection block (Linear→ReLU→Dropout) before
   the attention and classification convolutions. This gives the head a
   nonlinear transformation of backbone features before computing attention
   weights, increasing representational capacity.

2. Uses Conv1d (kernel_size=1) instead of Linear for att/cls projections.
   Mathematically equivalent when applied to (B, C, T) tensors, but Conv1d
   operates on the channel dimension natively in (B, C, T) format, avoiding
   the permute-back-permute pattern.

3. The backbone retains conv_head (the final 1x1 conv before pooling in
   EfficientNet), giving slightly more parameters (6.3M vs 4.6M).

This architecture scores 0.862-0.872 on the public LB when pretrained from
Perch (bioacoustic foundation model) and finetuned on BirdCLEF data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GEMFreqPool(nn.Module):
    """GEM pooling over frequency axis. Identical to v1."""
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)  # pool frequency
        return x.pow(1.0 / p)


class AttentionSEDHead(nn.Module):
    """
    SED head with FC projection + Conv1d attention/classification.

    Architecture: features → FC block → parallel Conv1d branches → attention pool

    The FC block (Linear→ReLU→Dropout) acts as a nonlinear bottleneck that
    transforms backbone features before they're split into attention and
    classification branches. This is more expressive than directly projecting
    raw features to class scores.
    """
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        # Nonlinear projection before attention/classification
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Attention and classification as Conv1d (kernel=1)
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T) from GEM pooling
        # FC block operates on features per time step
        x = x.permute(0, 2, 1)   # (B, T, C)
        x = self.fc(x)            # (B, T, C)
        x = x.permute(0, 2, 1)   # (B, C, T)

        att = torch.tanh(self.att_conv(x))   # (B, num_classes, T)
        att = F.softmax(att, dim=-1)         # normalize over time
        cls = self.cls_conv(x)               # (B, num_classes, T)

        clip_logits = (att * cls).sum(dim=-1)  # (B, num_classes)
        clip_probs = torch.sigmoid(clip_logits)

        return {
            "clipwise_logit": clip_logits,
            "clipwise_prob": clip_probs,
            "framewise_logit": cls.permute(0, 2, 1),  # (B, T, num_classes)
        }


class SEDModelV2(nn.Module):
    """
    SED model matching the public baseline architecture.

    Compatible with checkpoints from:
    - aidensong123/perch-fold (LB 0.862)
    - tonylica/birdclef-2026-model (LB 0.862, 0.872)
    """
    def __init__(self, backbone="tf_efficientnet_b0.ns_jft_in1k",
                 num_classes=234, in_channels=3, dropout=0.1,
                 gem_p_init=3.0, drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=False,
            in_chans=in_channels,
            features_only=False,
            global_pool="",
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        self.gem_pool = GEMFreqPool(p_init=gem_p_init)
        self.head = AttentionSEDHead(feat_dim, num_classes, dropout)

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.gem_pool(features)
        return self.head(pooled)
