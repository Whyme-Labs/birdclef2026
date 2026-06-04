"""
Audio Masked Autoencoder (Audio-MAE)

Self-supervised pretraining on unlabeled Pantanal soundscapes.
Masks 75% of mel spectrogram patches, trains ViT encoder to reconstruct.

Architecture:
    Mel spectrogram (1, 224, 320) → 16x16 patches → 280 tokens
    → Random mask 75% → Encoder (ViT-S, 384-dim) processes 70 visible
    → Decoder (lightweight, 192-dim) reconstructs 210 masked patches
    → MSE loss on masked patches only

References:
    - He et al., "Masked Autoencoders Are Scalable Vision Learners" (2022)
    - Huang et al., "Masked Autoencoders that Listen" (2022)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PatchEmbed(nn.Module):
    """Convert mel spectrogram to patch embeddings."""
    def __init__(self, img_size=(224, 320), patch_size=16, in_chans=1, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AudioMAE(nn.Module):
    """
    Audio Masked Autoencoder.

    Encoder: ViT-Small (12 layers, 384 dim, 6 heads) — processes visible patches only
    Decoder: Lightweight (4 layers, 192 dim, 3 heads) — reconstructs masked patches
    """
    def __init__(
        self,
        img_size=(224, 320),
        patch_size=16,
        in_chans=1,
        # Encoder
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        # Decoder
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4.,
        mask_ratio=0.75,
        norm_pix_loss=True,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, encoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, encoder_embed_dim))
        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio) for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans)

        self.initialize_weights()

    def initialize_weights(self):
        # Sinusoidal positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        dec_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """Random masking: keep (1-mask_ratio) of patches."""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Binary mask: 0=keep, 1=mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """Encode visible patches only."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]  # No cls token pos yet

        # Mask
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Encode
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Decode: insert mask tokens, reconstruct all patches."""
        x = self.decoder_embed(x)

        # Insert mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # Skip cls
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Prepend cls

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        return x[:, 1:, :]  # Remove cls

    def patchify(self, imgs):
        """Convert image to patches for loss computation."""
        p = self.patch_size
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(imgs.shape[0], 1, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(imgs.shape[0], h * w, p * p)
        return x

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Per-patch MSE
        loss = (loss * mask).sum() / mask.sum()  # Only masked patches

        return loss, pred, mask

    def encode(self, imgs):
        """Extract encoder features (for downstream tasks)."""
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x  # (B, num_patches+1, embed_dim)


class AudioMAEClassifier(nn.Module):
    """
    Classification model using pretrained Audio MAE encoder.

    Options:
    1. Global average pooling + linear head
    2. CLS token + linear head
    3. Attention pooling + SED head (for frame-level predictions)
    """
    def __init__(self, mae_encoder, num_classes=234, pool='avg', dropout=0.1):
        super().__init__()
        self.patch_embed = mae_encoder.patch_embed
        self.cls_token = mae_encoder.cls_token
        self.pos_embed = mae_encoder.pos_embed
        self.encoder_blocks = mae_encoder.encoder_blocks
        self.encoder_norm = mae_encoder.encoder_norm
        self.embed_dim = mae_encoder.pos_embed.shape[-1]

        self.pool = pool
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(self, imgs):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        if self.pool == 'cls':
            x = x[:, 0]
        else:  # avg pool over patch tokens
            x = x[:, 1:].mean(dim=1)

        logits = self.head(x)
        return {
            "clipwise_logit": logits,
            "clipwise_prob": torch.sigmoid(logits),
        }


# ── Positional embedding utilities ──────────────────────────────────────
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sinusoidal positional embeddings."""
    import numpy as np
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, gh, gw)
    grid = np.stack(grid, axis=0).reshape(2, -1)  # (2, gh*gw)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb = np.concatenate([emb_h, emb_w], axis=1)

    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    import numpy as np
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
