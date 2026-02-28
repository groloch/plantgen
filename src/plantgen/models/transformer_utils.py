import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(
        x: torch.Tensor,
        embedding_dim: int,
        frequency: float = 10000.0
    ) -> torch.Tensor:
    """
    Generate sinusoidal embedding for timesteps.

    Args:
        x: Tensor of shape (batch_size, ...) containing timestep values
        embedding_dim: Dimension of the output embeddings

    Returns:
        Embeddings of shape (batch_size, embedding_dim)
    """
    device = x.device
    dtype = x.dtype

    half_dim = embedding_dim // 2
    freqs = torch.exp(-torch.arange(half_dim, dtype=dtype, device=device) *
                      (torch.log(torch.tensor(frequency, dtype=dtype, device=device)) / half_dim))
    args = x[:, None] * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class SinePositionalEncoding1D(nn.Module):
    def __init__(self, frequency: float = 10000.0):
        super().__init__()
        self.frequency = frequency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        device = x.device
        dtype = x.dtype

        positions = torch.arange(N, device=device)

        freqs = torch.exp(
            torch.arange(0, D, 2, dtype=dtype, device=device) *
            -(math.log(self.frequency) / D)
        )

        pos_embeds = torch.zeros(N, D, dtype=dtype, device=device)
        pos_embeds[:, 0::2] = torch.cos(positions[:, None] * freqs[None, :])
        pos_embeds[:, 1::2] = torch.sin(positions[:, None] * freqs[None, :])

        pos_embeds = pos_embeds.unsqueeze(0).expand(B, N, D)

        return pos_embeds


class SinePositionalEncoding2D(nn.Module):
    """
    2D Sinusoidal positional encoding for spatial latents.
    Encodes both height and width dimensions separately.
    """
    def __init__(self, frequency: float = 10000.0):
        super().__init__()
        self.frequency = frequency

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, H*W, D) where H*W = N
            height: Height of the 2D spatial grid
            width: Width of the 2D spatial grid

        Returns:
            2D positional embeddings of shape (B, H*W, D)
        """
        B, N, D = x.shape
        device = x.device
        dtype = x.dtype

        assert N == height * width, f"Sequence length {N} must equal height*width {height*width}"
        assert D % 4 == 0, f"Embedding dimension {D} must be divisible by 4 for 2D encoding"

        half_dim = D // 2

        freqs = torch.exp(
            torch.arange(0, half_dim, 2, dtype=dtype, device=device) *
            -(math.log(self.frequency) / half_dim)
        )

        y_pos = torch.arange(height, dtype=dtype, device=device)
        x_pos = torch.arange(width, dtype=dtype, device=device)

        y_emb = torch.zeros(height, half_dim, dtype=dtype, device=device)
        y_emb[:, 0::2] = torch.cos(y_pos[:, None] * freqs[None, :])
        y_emb[:, 1::2] = torch.sin(y_pos[:, None] * freqs[None, :])

        x_emb = torch.zeros(width, half_dim, dtype=dtype, device=device)
        x_emb[:, 0::2] = torch.cos(x_pos[:, None] * freqs[None, :])
        x_emb[:, 1::2] = torch.sin(x_pos[:, None] * freqs[None, :])

        y_emb_grid = y_emb[:, None, :].expand(height, width, half_dim)
        x_emb_grid = x_emb[None, :, :].expand(height, width, half_dim)
        pos_embeds = torch.cat([y_emb_grid, x_emb_grid], dim=-1)

        pos_embeds = pos_embeds.reshape(N, D)
        pos_embeds = pos_embeds.unsqueeze(0).expand(B, N, D)

        return pos_embeds


class DiTModel(nn.Module):
    def forward(self, x, timestep, cond, attn_mask=None):
        raise NotImplementedError()
