"""
ProtoSSM v5 — Prototypical Selective State Space Model for BirdCLEF 2026.

Architecture overview:
─────────────────────
  Perch v2 embeddings (B, T=12, 1536)
    ↓
  Input projection: Linear(1536 → d_model) + LayerNorm + GELU
    ↓
  Learnable positional encoding (T=12)
    ↓
  Metadata injection: site_embed + hour_embed (additive)
    ↓
  N × Bidirectional SelectiveSSM layers
    ↓
  TemporalCrossAttention (multi-head self-attention)
    ↓
  Prototypical classification: cosine_sim(h, prototypes) × τ + bias
    ↓
  Gated fusion: σ(α) × proto_score + (1-σ(α)) × perch_logits

Mathematical foundations:
─────────────────────────
1. Selective SSM (Mamba-style):
   Continuous: dx/dt = Ax + Bu,  y = Cx + Du
   Discretized: x_t = Ā·x_{t-1} + B̄·u_t,  y_t = C·x_t + D·u_t
   where Ā = exp(Δ·A),  B̄ ≈ Δ·B

   "Selective" means Δ, B, C are input-dependent:
     Δ_t = softplus(Linear(u_t))   — input-dependent step size
     B_t = Linear(u_t)              — input-dependent input projection
     C_t = Linear(u_t)              — input-dependent readout

   HiPPO A initialization: A_i = -exp(log(i)), i=1,...,d_state
   This gives exponentially spaced decay rates for multi-scale memory.

2. Bidirectional scan:
   h_fwd = SSM_scan(x_1, ..., x_T)
   h_bwd = SSM_scan(x_T, ..., x_1)
   h = Linear(concat(h_fwd, h_bwd))

3. Prototypical classification:
   For class c with K prototypes {p_{c,k}}:
     s_c(h) = (1/K) Σ_k cos(h, p_{c,k}) × τ + b_c
   where τ is learnable temperature, b_c is per-class bias.

4. Residual boosting (ResBoosting):
   r = y - σ(ŷ^(1))                    — residuals from first pass
   ŷ^(2) = f_res([x, ŷ^(1)])          — 2nd SSM on residuals
   ŷ_final = ŷ^(1) + w × ŷ^(2)        — additive correction

References:
  - Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
  - Snell et al., "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
  - Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (2001)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Selective SSM Layer
# ═══════════════════════════════════════════════════════════════════════════════

class SelectiveSSM(nn.Module):
    """
    Simplified Mamba-style selective state space model.

    Input-dependent discretization makes this a sequence-to-sequence model
    that can selectively attend to or ignore inputs based on content.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection: x → (z, x_inner) for gated output
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal conv1d for local context before SSM
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True
        )

        # SSM parameters: input-dependent B, C, delta
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable log(A) — HiPPO initialization
        # A_i = -exp(log(i+1)) gives multi-scale decay rates
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1).clone())

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # LayerNorm for stability
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, reverse=False):
        """
        Args:
            x: (B, T, D) input sequence
            reverse: if True, scan from T→1 (backward pass)
        Returns:
            (B, T, D) output sequence
        """
        residual = x
        x = self.norm(x)
        B, T, D = x.shape

        # Project and split into gated branches
        xz = self.in_proj(x)                           # (B, T, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)               # each (B, T, d_inner)

        # Causal conv1d
        x_inner = x_inner.transpose(1, 2)              # (B, d_inner, T)
        x_inner = self.conv1d(x_inner)[:, :, :T]       # causal: trim to T
        x_inner = x_inner.transpose(1, 2)              # (B, T, d_inner)
        x_inner = F.silu(x_inner)

        # Input-dependent SSM parameters
        x_proj = self.x_proj(x_inner)                  # (B, T, 2*d_state + 1)
        B_input = x_proj[:, :, :self.d_state]           # (B, T, d_state)
        C_input = x_proj[:, :, self.d_state:2*self.d_state]  # (B, T, d_state)
        delta = F.softplus(x_proj[:, :, -1])            # (B, T) — step size

        # Compute A (negative for stability)
        A = -torch.exp(self.A_log)                      # (d_inner, d_state)

        # Sequential scan
        if reverse:
            x_inner = x_inner.flip(1)
            B_input = B_input.flip(1)
            C_input = C_input.flip(1)
            delta = delta.flip(1)

        y = self._ssm_scan(x_inner, A, B_input, C_input, delta)

        if reverse:
            y = y.flip(1)

        # Gated output: y × SiLU(z)
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y + residual

    def _ssm_scan(self, u, A, B, C, delta):
        """
        Sequential SSM scan.

        Args:
            u: (B, T, d_inner) input
            A: (d_inner, d_state) state matrix
            B: (B, T, d_state) input-dependent B
            C: (B, T, d_state) input-dependent C
            delta: (B, T) step sizes
        Returns:
            (B, T, d_inner) output
        """
        B_batch, T, d_inner = u.shape
        d_state = A.shape[1]

        # Discretize: Ā = exp(Δ × A), B̄ = Δ × B
        # Clamp delta for numerical stability (avoid extremely large steps)
        delta = delta.clamp(max=5.0)
        delta_expand = delta.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        A_expand = A.unsqueeze(0).unsqueeze(0)            # (1, 1, d_inner, d_state)
        dA = torch.exp(delta_expand * A_expand)           # (B, T, d_inner, d_state)
        dA = dA.clamp(max=1.0)  # ensure stable (non-exploding) state transitions
        dB = delta.unsqueeze(-1).unsqueeze(-1) * B.unsqueeze(2)  # (B, T, 1, d_state)

        # State: (B, d_inner, d_state)
        h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs = []

        for t in range(T):
            # x_t = Ā·x_{t-1} + B̄·u_t
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            # y_t = C·x_t + D·u_t
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1) + self.D * u[:, t]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, T, d_inner)


class BidirectionalSSMLayer(nn.Module):
    """Bidirectional SSM: forward scan + backward scan → merge."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.fwd_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.bwd_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        h_fwd = self.fwd_ssm(x, reverse=False)
        h_bwd = self.bwd_ssm(x, reverse=True)
        return self.merge(torch.cat([h_fwd, h_bwd], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Attention
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalCrossAttention(nn.Module):
    """
    Post-norm multi-head self-attention over the temporal axis.
    Captures non-local patterns the sequential SSM misses
    (e.g., dawn chorus onset, counter-singing dynamics).
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Pre-norm FFN
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# ProtoSSM v5
# ═══════════════════════════════════════════════════════════════════════════════

class ProtoSSMv5(nn.Module):
    """
    Full ProtoSSM v5 model.

    Takes Perch v2 embeddings (B, T, 1536) + Perch logits (B, T, C) + metadata
    and outputs per-window species probabilities (B, T, C).
    """
    def __init__(
        self,
        emb_dim=1536,
        num_classes=234,
        d_model=320,
        d_state=32,
        n_ssm_layers=4,
        n_heads=8,
        n_prototypes=2,
        n_sites=30,
        n_hours=24,
        meta_dim=24,
        dropout=0.1,
        use_cross_attn=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_prototypes = n_prototypes
        self.use_cross_attn = use_cross_attn

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable positional encoding (T=12 windows in a 60s file)
        self.pos_enc = nn.Parameter(torch.randn(1, 12, d_model) * 0.02)

        # Metadata embeddings (site + hour)
        self.site_emb = nn.Embedding(n_sites + 1, meta_dim)  # +1 for unknown
        self.hour_emb = nn.Embedding(n_hours, meta_dim)
        self.meta_proj = nn.Linear(meta_dim * 2, d_model)

        # Bidirectional SSM layers
        self.ssm_layers = nn.ModuleList([
            BidirectionalSSMLayer(d_model, d_state, d_conv=4, expand=2)
            for _ in range(n_ssm_layers)
        ])

        # Cross-attention after SSM
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(d_model, n_heads, dropout)

        self.final_norm = nn.LayerNorm(d_model)

        # Prototypical classification head
        # K learnable prototypes per class, cosine similarity
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, n_prototypes, d_model) * 0.02
        )
        self.temperature = nn.Parameter(torch.tensor(10.0))  # learnable τ
        self.class_bias = nn.Parameter(torch.zeros(num_classes))

        # Gated fusion with Perch logits: per-class learnable α
        # α > 0 → trust ProtoSSM more; α < 0 → trust Perch more
        self.fusion_alpha = nn.Parameter(torch.zeros(num_classes))

        # Taxonomic auxiliary head (optional regularizer)
        self.aux_head = nn.Linear(d_model, num_classes)

    def forward(self, embeddings, perch_logits, site_ids, hour_ids):
        """
        Args:
            embeddings: (B, T, 1536) Perch v2 embeddings
            perch_logits: (B, T, C) raw Perch v2 logits (mapped to competition species)
            site_ids: (B,) recording site index
            hour_ids: (B,) hour of day (0-23)
        Returns:
            dict with 'logits' (B, T, C), 'aux_logits' (B, T, C)
        """
        B, T, _ = embeddings.shape

        # Project embeddings
        x = self.input_proj(embeddings)  # (B, T, d_model)

        # Add positional encoding
        x = x + self.pos_enc[:, :T, :]

        # Inject metadata
        site_e = self.site_emb(site_ids)      # (B, meta_dim)
        hour_e = self.hour_emb(hour_ids)      # (B, meta_dim)
        meta = self.meta_proj(torch.cat([site_e, hour_e], dim=-1))  # (B, d_model)
        x = x + meta.unsqueeze(1)             # broadcast over time

        # SSM layers
        for ssm_layer in self.ssm_layers:
            x = ssm_layer(x)

        # Cross-attention
        if self.use_cross_attn:
            x = self.cross_attn(x)

        x = self.final_norm(x)  # (B, T, d_model)

        # Prototypical classification
        # x: (B, T, d_model), prototypes: (C, K, d_model)
        # Normalize for cosine similarity
        x_norm = F.normalize(x, dim=-1)                                    # (B, T, D)
        proto_norm = F.normalize(self.prototypes, dim=-1)                   # (C, K, D)

        # Cosine similarity: (B, T, C) = mean over K prototypes
        # Reshape for batch matmul: x_norm (B*T, D) @ proto_norm (C*K, D).T → (B*T, C*K)
        BT = B * T
        sim = torch.mm(
            x_norm.reshape(BT, -1),
            proto_norm.reshape(-1, self.d_model).T
        )  # (BT, C*K)
        sim = sim.reshape(BT, self.num_classes, self.n_prototypes)  # (BT, C, K)
        sim = sim.mean(dim=-1)  # (BT, C) — average over prototypes
        sim = sim.reshape(B, T, self.num_classes)

        # Scale by temperature + add bias
        # Clamp temperature to prevent exploding logits
        tau = self.temperature.clamp(min=1.0, max=50.0)
        proto_logits = sim * tau + self.class_bias

        # Gated fusion with Perch logits
        alpha = torch.sigmoid(self.fusion_alpha)  # (C,) in [0, 1]
        logits = alpha * proto_logits + (1 - alpha) * perch_logits

        # Auxiliary head for regularization
        aux_logits = self.aux_head(x)  # (B, T, C)

        return {
            'logits': logits,
            'proto_logits': proto_logits,
            'aux_logits': aux_logits,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Residual SSM (ResBoosting)
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualSSM(nn.Module):
    """
    Second-pass error-correction SSM.

    Trained on residuals r = y - σ(ŷ^(1)) from the first-pass ensemble.
    Takes BOTH embeddings and first-pass predictions as input, enabling
    the model to learn corrections conditioned on both what was "heard"
    (embeddings) and what was "predicted" (first-pass scores).

    Mathematical basis:
      This is equivalent to one step of functional gradient descent:
        ŷ^(k+1) = ŷ^(k) + η · ∇L(y, ŷ^(k))
      where the gradient is approximated by a neural network f_res.
      This is the neural analog of gradient boosting (Friedman, 2001).
    """
    def __init__(
        self,
        emb_dim=1536,
        num_classes=234,
        d_model=128,
        d_state=16,
        n_ssm_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        # Project: concat(embeddings, first_pass_scores) → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(emb_dim + num_classes, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Bidirectional SSM layers
        self.ssm_layers = nn.ModuleList([
            BidirectionalSSMLayer(d_model, d_state, d_conv=4, expand=2)
            for _ in range(n_ssm_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output head — initialized near zero so corrections start small
        self.out_head = nn.Linear(d_model, num_classes)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, embeddings, first_pass_scores):
        """
        Args:
            embeddings: (B, T, 1536) Perch v2 embeddings
            first_pass_scores: (B, T, C) first-pass logits (pre-sigmoid)
        Returns:
            corrections: (B, T, C) additive corrections
        """
        x = torch.cat([embeddings, first_pass_scores.detach()], dim=-1)
        x = self.input_proj(x)

        for ssm_layer in self.ssm_layers:
            x = ssm_layer(x)

        x = self.norm(x)
        return self.out_head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# MLP Probe (lightweight per-class classifier)
# ═══════════════════════════════════════════════════════════════════════════════

class MLPProbe(nn.Module):
    """
    Per-class MLP probe on PCA-compressed embeddings + sequential features.

    This is a simple feedforward classifier that operates on engineered
    features rather than raw embeddings. It provides a different inductive
    bias from the SSM and serves as an ensemble member.
    """
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
