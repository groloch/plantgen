import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.models.dit_config import MMDiTConfig
from .transformer_utils import (
    SinePositionalEncoding2D,
    get_timestep_embedding,
    DiTModel
)


class SelfAttention(DiTModel):
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()

        self.x_norm = nn.LayerNorm(hidden_dim)
        self.c_norm = nn.LayerNorm(hidden_dim)

        self.inx_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.inc_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.outx_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.outc_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.mx_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.sx_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.ax_proj = nn.Linear(hidden_dim, 1, bias=False)

        self.mc_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.sc_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.ac_proj = nn.Linear(hidden_dim, 1, bias=False)

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

    def forward(
            self,
            x: torch.Tensor,
            c: torch.Tensor,
            y: torch.Tensor,
            attn_mask: torch.Tensor = None
    ):
        B, N, D = x.shape
        _, M, _ = c.shape

        x_ = self.x_norm(x)
        x_ = self.mx_proj(y) * x_ + self.sx_proj(y)
        x_ = self.inx_proj(x_)

        c_ = self.c_norm(c)
        c_ = self.mc_proj(y) * c_ + self.sc_proj(y)
        c_ = self.inc_proj(c_)

        cx = torch.cat([x_, c_], dim=1)

        q = self.q_proj(cx).view(B, N + M, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(cx).view(B, N + M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(cx).view(B, N + M, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        attn = attn.transpose(1, 2).reshape(B, N+M, D)
        attn = self.o_proj(attn)

        x_ = self.outx_proj(attn[:, :N, :]) * self.ax_proj(y)
        c_ = self.outc_proj(attn[:, N:, :]) * self.ac_proj(y)

        return x + x_, c + c_


class FFN(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.x_norm = nn.LayerNorm(hidden_dim)
        self.c_norm = nn.LayerNorm(hidden_dim)

        self.x_fc1 = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.x_fc2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
        self.c_fc1 = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.c_fc2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

        self.mx_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.sx_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.ax_proj = nn.Linear(hidden_dim, 1, bias=False)

        self.mc_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.sc_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.ac_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, c, y):
        x_ = self.x_norm(x)
        x_ = self.mx_proj(y) * x_ + self.sx_proj(y)
        x_ = F.gelu(self.x_fc1(x_))
        x_ = self.x_fc2(x_)
        x_ = self.ax_proj(y) * x_

        c_ = self.c_norm(c)
        c_ = self.mc_proj(y) * c_ + self.sc_proj(y)
        c_ = F.gelu(self.c_fc1(c_))
        c_ = self.c_fc2(c_)
        c_ = self.ac_proj(y) * c_

        return x + x_, c + c_


class MMDiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()

        self.attn = SelfAttention(hidden_dim, n_heads)
        self.ffn = FFN(hidden_dim)

    def forward(
            self,
            x: torch.Tensor,
            c: torch.Tensor,
            y: torch.Tensor,
            attn_mask: torch.Tensor = None
    ):
        x, c = self.attn(x, c, y, attn_mask)
        x, c = self.ffn(x, c, y)
        return x, c


class MMDiT(DiTModel):
    """
    MM-DiT (MultiModal Diffusion Transformer) implementation similar to the one introduced 
    by StableDiffusion3 (https://arxiv.org/pdf/2403.03206)

    Main difference is that we don't combine the pooled text embeddings with the time embeddings
    for the y vector, but only use the time embeddings as we've had better results with this approach.
    """
    def __init__(self, config: MMDiTConfig):
        super().__init__()


        self.in_proj = nn.Linear(
            config.latent_dim,
            config.hidden_dim
        )
        self.text_in_proj = nn.Linear(
            config.text_embed_dim,
            config.hidden_dim
        )

        self.time_y_proj = nn.Linear(
            config.hidden_dim,
            config.hidden_dim
        )

        self.layers = nn.ModuleList([
            MMDiTBlock(
                hidden_dim=config.hidden_dim,
                n_heads=config.n_heads,
            ) for _ in range(config.n_layers)
        ])

        self.out_norm = nn.LayerNorm(config.hidden_dim)
        self.out_proj = nn.Linear(
            config.hidden_dim,
            config.latent_dim
        )

        self.positional_encoding = SinePositionalEncoding2D(
            frequency=config.sine_encoding_frequency
        )

        self.hidden_dim = config.hidden_dim

    def forward(self, x, timestep, embeds=None, attn_mask=None):
        B, C, W, H = x.shape
        N = W * H

        x = x.permute(0, 2, 3, 1).reshape(B, N, -1)
        x = self.in_proj(x)
        x = x + self.positional_encoding(x, W, H)

        c = self.text_in_proj(embeds) # B M D

        t_embd = get_timestep_embedding(
            timestep.view(B, -1),
            self.hidden_dim
        )

        y = self.time_y_proj(t_embd)

        if embeds is not None and attn_mask is not None:
            M = attn_mask.size(1)
            N_ = N + c.shape[1] - M
            attn_mask = attn_mask.bool()

            # top left
            latent_to_latent = torch.ones(
                B, 1, N_, N_,
                device=x.device, dtype=torch.bool
            )

            # top right
            latent_to_text = attn_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, N_, M)

            # bottom left
            text_to_latent = attn_mask.unsqueeze(1).unsqueeze(3).expand(B, 1, M, N_)

            # bottom right
            text_to_text = attn_mask.unsqueeze(1) & attn_mask.unsqueeze(2)
            text_to_text = text_to_text.unsqueeze(1)

            top_row = torch.cat([latent_to_latent, latent_to_text], dim=-1)
            bottom_row = torch.cat([text_to_latent, text_to_text], dim=-1)
            attn_mask = torch.cat([top_row, bottom_row], dim=2)

            attn_mask = attn_mask.expand(B, self.layers[0].attn.n_heads, N_ + M, N_ + M)
        else:
            attn_mask = None

        for layer in self.layers:
            x, c = layer(x, c, y, attn_mask=attn_mask)

        x = self.out_norm(x)
        x = self.out_proj(x)
        x = x.view(B, W, H, C).permute(0, 3, 1, 2)

        return x


if __name__ == "__main__":
    from ..config.models.dit_config import MMDiTConfig
    from ..utils import model_parameters

    config = MMDiTConfig(
        latent_dim = 16,
        latent_size = 64,
        hidden_dim = 128,
        n_heads = 8,
        n_layers = 10,
        text_embed_dim = 768,
        patch_size=8,
        sine_encoding_frequency=10000.0
    )
    model = MMDiT(config)

    params, _ = model_parameters(model)
    print(f'{params:,} parameters')

    x = torch.randn(4, 16, 8, 8)
    timestep = torch.rand(4, 1, 1, 1)

    from transformers import AutoTokenizer, AutoModel
    text_encoder = AutoModel.from_pretrained('google/embeddinggemma-300m')
    tokenizer = AutoTokenizer.from_pretrained('google/embeddinggemma-300m')
    sentences = ["A photo of a cat that is eating a cake.", "A photo of a dog."]*2

    with torch.no_grad():
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    embed = text_encoder(**inputs).last_hidden_state
    attn_mask = inputs['attention_mask']

    out = model(x, timestep, embed, attn_mask=attn_mask)
    print(out.shape)
