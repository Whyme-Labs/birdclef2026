"""
Path 2 architecture: Multi-Window SED with cross-window attention.

Key reframing vs single-window SED:
  Single-window: P(species | mel_t) — independent per 5s window
  Multi-window:  P(species_t | mel_1, ..., mel_W) — joint over the 60s file

The mathematical insight is that file-level evidence (a species heard in window 5
makes it more probable in adjacent windows; site/time priors aggregate across
windows) is information the per-window model cannot use. Cross-window self-
attention before the classifier head lets every window's prediction be informed
by features extracted from every other window in the same recording.

Architecture:
  (B, W, 3, M, T)
  → per-window backbone + GEM → (B*W, T', C_feat)
  → time-pool to per-window summary (B, W, C_feat)
  → cross-window self-attention (B, W, C_feat)
  → broadcast back & residual into per-window time-frame features
  → original AttentionHead → (B, W, num_classes), (B, W, T', num_classes)
"""
import torch
import torch.nn as nn
import timm

from src.models import GEMPool2d, AttentionHead


class CrossWindowAttn(nn.Module):
    """
    Pre-norm transformer encoder over the W-window axis with bottleneck projection.

    Architecture:
      x → proj_in (d_model→d_inner) + pos_emb → encoder → LayerNorm
        → proj_out (d_inner→d_model, small-gain init)

    The LayerNorm bounds the output magnitude regardless of encoder weights, and
    the small-gain proj_out keeps initial injection to ~10% of the feature scale.
    Both cw_attn weights AND the residual gate (in MultiWindowSED) receive
    gradient signal from step 0 — no chicken-and-egg.
    """
    def __init__(self, d_model, n_windows=12, d_inner=256, n_heads=4,
                 n_layers=1, dropout=0.1, out_gain=0.1):
        super().__init__()
        self.proj_in = nn.Linear(d_model, d_inner)
        self.pos_emb = nn.Parameter(torch.randn(1, n_windows, d_inner) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_inner, nhead=n_heads,
            dim_feedforward=d_inner * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_inner)
        self.proj_out = nn.Linear(d_inner, d_model)
        # Small-gain init: initial injection ~10% of feature scale, harmless but
        # nonzero so all cw_attn weights get gradient from step 0.
        nn.init.xavier_uniform_(self.proj_out.weight, gain=out_gain)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):  # (B, W, D)
        h = self.proj_in(x) + self.pos_emb
        h = self.encoder(h)
        h = self.norm(h)
        return self.proj_out(h)


class MultiWindowSED(nn.Module):
    """
    Drop-in replacement for SEDModel that processes W windows jointly.

    Forward:
      x: (B, W, 3, n_mels, T) → (B, W, num_classes), (B, W, T', num_classes)

    The cross-window attention contribution is gated by a learnable scalar
    initialized to zero, so the model boots up identical to the single-window
    SEDModel and only adds joint reasoning if the gradient says it helps.
    """
    def __init__(self, config, n_windows=12, cw_heads=4, cw_layers=2, cw_dropout=0.1):
        super().__init__()
        self.config = config
        self.n_windows = n_windows

        self.backbone = timm.create_model(
            config.backbone, pretrained=True, in_chans=config.in_chans,
            num_classes=0, global_pool="",
            drop_rate=0.0, drop_path_rate=0.0,
        )
        with torch.no_grad():
            dummy = torch.randn(1, config.in_chans, config.n_mels, 313)
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[1]

        self.gem = GEMPool2d(p_init=config.gem_p_init,
                             trainable=config.gem_p_trainable, dim=2)
        self.dropout = nn.Dropout(config.drop_rate)

        self.cw_attn = CrossWindowAttn(
            d_model=self.feat_dim, n_windows=n_windows,
            n_heads=cw_heads, n_layers=cw_layers, dropout=cw_dropout,
            out_gain=0.1,
        )
        # Gate init=1.0 with cw_attn out_gain=0.1: initial injection magnitude
        # is ~10% of feature scale (small but nonzero), so all cw_attn weights
        # AND the gate get gradient from step 0. The gate can grow if joint
        # reasoning helps and shrink toward 0 if cw_attn is harmful.
        self.cw_gate = nn.Parameter(torch.tensor(1.0))

        self.head = AttentionHead(self.feat_dim, config.num_classes)

    def forward(self, x):
        # x: (B, W, 3, n_mels, T)
        if x.dim() == 4:                      # single-window fallback
            B, _, _, _ = x.shape
            x = x.unsqueeze(1)                # (B, 1, 3, M, T)
        B, W, C, M, T = x.shape
        x = x.reshape(B * W, C, M, T)

        feat = self.backbone(x)               # (B*W, C', F', T')
        feat = self.gem(feat)                 # (B*W, C', T')
        feat = feat.permute(0, 2, 1)          # (B*W, T', C')
        Tprime = feat.shape[1]

        # Per-window summary (mean over time) → cross-window attention
        win_summary = feat.mean(dim=1).reshape(B, W, -1)
        cw_out = self.cw_attn(win_summary)    # (B, W, C')

        # Residual injection: broadcast cw_out across time, gate by cw_gate
        injection = cw_out.unsqueeze(2).expand(-1, -1, Tprime, -1).reshape(B * W, Tprime, -1)
        feat = feat + self.cw_gate * injection

        feat = self.dropout(feat)
        clip_logits, frame_logits = self.head(feat)

        clip_logits = clip_logits.reshape(B, W, -1)
        frame_logits = frame_logits.reshape(B, W, frame_logits.shape[-2], frame_logits.shape[-1])
        return clip_logits, frame_logits

    def load_single_window_state(self, state_dict, strict_backbone=True):
        """
        Warm-start from a single-window SEDModel checkpoint.
        Loads backbone, gem, head; cw_attn / cw_gate stay at init (gate=0 → identity).
        """
        own = self.state_dict()
        loaded = 0
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        self.load_state_dict(own)
        return loaded

    def get_param_groups(self, config, cw_lr_mult=4.0):
        """
        Three-group setup so the newly-added cross-window attention can develop
        faster than the warm-started head/backbone:
          - backbone: lr * backbone_lr_mult (e.g., 0.1)
          - head/gem (warm-started): lr
          - cw_attn / cw_gate (fresh init): lr * cw_lr_mult (e.g., 4.0)
        """
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.gem.parameters())
            + list(self.head.parameters())
            + list(self.dropout.parameters())
        )
        cw_params = list(self.cw_attn.parameters()) + [self.cw_gate]
        return [
            {"params": backbone_params, "lr": config.lr * config.backbone_lr_mult},
            {"params": head_params, "lr": config.lr},
            {"params": cw_params, "lr": config.lr * cw_lr_mult},
        ]
