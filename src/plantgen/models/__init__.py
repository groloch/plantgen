from .conv import *
from .vae import ConvVAE
from .cross_dit import CrossDIT
from .mm_dit import MMDiT
from .transformer_utils import DiTModel

from ..config.models import DiTConfig


def build_dit_model(config: DiTConfig) -> DiTModel:
    if config.model_type == 'crossdit':
        model = CrossDIT(config)
    elif config.model_type == 'mmdit':
        model = MMDiT(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model
