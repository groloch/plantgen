import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_utils import (
    get_timestep_embedding,
    SinePositionalEncoding1D,
    SinePositionalEncoding2D,
    DiTModel
)
from ..config.models.dit_config import CrossDITConfig


class FFN(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        self.m_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.s_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.a_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, t_embd):
        x_ = self.norm(x)
        x_ = self.m_proj(t_embd) * x_ + self.s_proj(t_embd)

        x_ = self.fc1(x_)
        x_ = F.gelu(x_)
        x_ = self.fc2(x_)

        x_ = x_ * self.a_proj(t_embd)

        return x + x_


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, latent_size: int, n_heads: int):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.m_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.s_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.a_proj = nn.Linear(hidden_dim, 1, bias=False)

        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

    def forward(self, x, t_embd, pos_embd):
        B, N, D = x.shape

        x_ = self.norm(x)
        x_ = self.m_proj(t_embd) * x_ + self.s_proj(t_embd)
        x_ = x_ + pos_embd

        q = self.q_proj(x_)  # (B, N, D)
        k = self.k_proj(x_)  # (B, N, D)
        v = self.v_proj(x_)  # (B, N, D)

        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, D)
        attn = self.o_proj(attn) * self.a_proj(t_embd)

        return x + attn


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, latent_size: int):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.m_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.s_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.a_proj = nn.Linear(hidden_dim, 1, bias=False)

        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.latent_size = latent_size

    def forward(
            self,
            x,
            t_embd,
            pos_embd,
            cond,
            attn_mask=None
        ):
        B, N, D = x.shape
        _, M, _ = cond.shape

        x_ = self.norm(x)
        x_ = self.m_proj(t_embd) * x_ + self.s_proj(t_embd)
        x_ = x_ + pos_embd

        q = self.q_proj(x_)  # (B, N, D)
        k = self.k_proj(cond)  # (B, M, D)
        v = self.v_proj(cond)  # (B, M, D)

        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = k.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, M, d)
        v = v.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, M, d)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask
        )

        attn = attn.transpose(1, 2).reshape(B, N, D)
        attn = self.o_proj(attn) * self.a_proj(t_embd)

        return x + attn


class CrossDITBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            n_heads: int,
            latent_size: int):
        super().__init__()

        self.self_attn = SelfAttention(hidden_dim, latent_size, n_heads)
        self.cross_attn = CrossAttention(hidden_dim, n_heads, latent_size)
        self.ffn = FFN(hidden_dim, hidden_dim * 4)

        self.latent_size = latent_size

    def forward(self, x, t_embd, pos_embd, cond, attn_mask=None):
        x = self.self_attn(x, t_embd, pos_embd)
        x = self.cross_attn(x, t_embd, pos_embd, cond, attn_mask=attn_mask)
        x = self.ffn(x, t_embd)
        return x


class CrossDIT(DiTModel):
    def __init__(self, config: CrossDITConfig):
        super().__init__()

        self.in_proj = nn.Linear(
            config.latent_dim,
            config.hidden_dim
        )
        self.text_in_proj = nn.Linear(
            config.text_embed_dim,
            config.hidden_dim
        )

        self.layers = nn.ModuleList([
            CrossDITBlock(
                hidden_dim=config.hidden_dim,
                n_heads=config.n_heads,
                latent_size=config.latent_size ** 2,
            ) for _ in range(config.n_layers)
        ])

        self.n_heads = config.n_heads

        self.out_proj = nn.Linear(
            config.hidden_dim,
            config.latent_dim
        )

        self.hidden_dim = config.hidden_dim
        self.latent_size = config.latent_size

        self.positional_encoding = SinePositionalEncoding2D(
            frequency=config.sine_encoding_frequency
        )

    def forward(self, x, timestep, cond, attn_mask=None):
        B, C, W, H = x.shape
        N = W * H

        x = x.permute(0, 2, 3, 1).reshape(B, N, -1)
        x = self.in_proj(x) # (B, N, D)

        cond = self.text_in_proj(cond)  # (B, L, D)

        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).expand(
                B,
                self.n_heads,
                N,
                attn_mask.size(1)
            )

        t_embd = get_timestep_embedding(
            timestep.view(B, -1),
            self.hidden_dim
        )  # (B, 1, D)

        pos_embd = self.positional_encoding(x, self.latent_size, self.latent_size)

        for layer in self.layers:
            x = layer(x, t_embd, pos_embd, cond, attn_mask=attn_mask)

        x = self.out_proj(x)
        x = x.reshape(B, W, H, C).permute(0, 3, 1, 2)

        return x


if __name__ == "__main__":
    config = CrossDITConfig(
        latent_dim = 16,
        latent_size = 64,
        hidden_dim = 1024,
        n_heads = 8,
        n_layers = 10,
        text_ctx_len = 1024,
        text_embed_dim = 768,
        patch_size=8
    )

    model = CrossDIT(config)

    x = torch.randn(2, 16, 64, 64)

    from transformers import AutoTokenizer, AutoModel
    text_encoder = AutoModel.from_pretrained('google/embeddinggemma-300m')
    tokenizer = AutoTokenizer.from_pretrained('google/embeddinggemma-300m')
    sentences = ["A photo of a cat that is eating a cake.", "A photo of a dog."]

    with torch.no_grad():
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    print(inputs.keys())
    embed = text_encoder(**inputs).last_hidden_state
    attn_mask = inputs['attention_mask']
    print(embed.shape, inputs['attention_mask'].shape)

    timestep = torch.rand(2, 1, 1, 1)

    out = model(x, timestep, embed, attn_mask)
    print(out.shape)
