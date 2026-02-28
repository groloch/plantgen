from dataclasses import dataclass


@dataclass
class ConvVAEConfig:
    in_channels: int
    image_size: int
    latent_dim: int
    depths: list[int]
    dims: list[int]
    ln_eps: float
    iaf: bool
    iaf_n_blocks: int = None
    iaf_timesteps: int = None
