from dataclasses import dataclass


@dataclass
class DiTConfig:
    model_type: str
    latent_dim: int
    latent_size: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    text_embed_dim: int
    patch_size: int
    sine_encoding_frequency: float
    num_classes: int


@dataclass
class CrossDITConfig(DiTConfig):
    pass

@dataclass
class MMDiTConfig(DiTConfig):
    pass