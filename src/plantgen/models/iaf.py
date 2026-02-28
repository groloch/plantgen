import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ConvNextBlock, ChannelFirstLayerNorm


class IAFSelfAttention(nn.Module):
    def __init__(self, latent_dim: int, ln_eps: float):
        super().__init__()

        self.q_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.k_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.v_proj = nn.Linear(latent_dim, latent_dim, bias=False)

        self.norm = nn.LayerNorm(latent_dim, eps=ln_eps)

    def forward(self, z):
        z = self.norm(z)
        q = self.q_proj(z)
        k = self.k_proj(z)
        v = self.v_proj(z)

        attn = F.scaled_dot_product_attention(q, k, v)

        return z + attn


class IAFCrossAttentionBlock(nn.Module):
    def __init__(self, latent_dim: int, ln_eps: float):
        super().__init__()

        self.q_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.k_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.v_proj = nn.Linear(latent_dim, latent_dim, bias=False)

        self.norm = nn.LayerNorm(latent_dim, eps=ln_eps)

    def forward(self, z, zp):
        z = self.norm(z)
        q = self.q_proj(z)
        k = self.k_proj(zp)
        v = self.v_proj(zp)

        attn = F.scaled_dot_product_attention(q, k, v)

        return z + attn


class IAFBlock(nn.Module):
    def __init__(self, latent_dim: int, latent_image_size: int, iaf_steps: int, ln_eps: float):
        super().__init__()

        self.self_attn = IAFSelfAttention(latent_dim, ln_eps)
        self.cross_attn = IAFCrossAttentionBlock(latent_dim, ln_eps)

        self.cnx_block = ConvNextBlock(dim=latent_dim, eps=ln_eps)

        self.spatial_pos_embed = nn.Parameter(
            torch.randn(latent_image_size*latent_image_size, latent_dim)
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(iaf_steps, latent_dim)
        )

    def forward(self, z: torch.Tensor, *zs: torch.Tensor):
        B, C, W, H = z.shape
        N = len(zs)

        zp = torch.stack(zs, dim=0)

        z = z.permute(0, 2, 3, 1).reshape(B, W*H, C)
        zp = zp.permute(0, 3, 4, 1, 2).reshape(B, W*H, N, -1) # B, W*H, N, C
        zp = zp + self.temporal_pos_embed[:N]

        zp = zp.permute(0, 2, 1, 3) # B, N, W*H, C
        zp = zp + self.spatial_pos_embed

        zp = zp.reshape(B, N*W*H, -1)

        z = self.self_attn(z+self.spatial_pos_embed)
        z = self.cross_attn(z+self.spatial_pos_embed, zp)

        z = z.reshape(B, W, H, C).permute(0, 3, 1, 2)

        z = self.cnx_block(z)

        return z


class IAFModel(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            latent_image_size: int,
            iaf_steps: int,
            ln_eps: float,
            n_blocks: int):
        super().__init__()

        self.in_proj = nn.Conv2d(latent_dim*2+1, latent_dim, kernel_size=1, padding='same')

        self.blocks = nn.ModuleList([
            IAFBlock(
                latent_dim,
                latent_image_size,
                iaf_steps,
                ln_eps
                ) for _ in range(n_blocks)
        ])

        self.m_proj = nn.Conv2d(latent_dim, latent_dim, kernel_size=1, padding='same')
        self.s_proj = nn.Conv2d(latent_dim, latent_dim, kernel_size=1, padding='same')

        nn.init.zeros_(self.m_proj.weight)
        nn.init.zeros_(self.m_proj.bias)
        nn.init.zeros_(self.s_proj.weight)
        nn.init.constant_(self.s_proj.bias, 1.0)

    def forward(
            self,
            z: torch.Tensor,
            h: torch.Tensor,
            t: torch.Tensor,
            *zs):
        B, C, W, H = z.shape

        x = torch.cat([z, h, t], dim=1) # (B, 2C+1, W, H)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, *zs)

        m = self.m_proj(x)
        s = self.s_proj(x)
        return m, s
