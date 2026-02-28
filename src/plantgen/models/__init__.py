from torchvision.models import resnet18, resnet50, efficientnet_b0

from .conv import *
from .vae import ConvVAE
from .cross_dit import CrossDIT
from .mm_dit import MMDiT
from .transformer_utils import DiTModel


MODEL_TYPE_DICT = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'efficientnet_b0': efficientnet_b0
}


def build_dit_model(config):
    if config.model_type == 'crossdit':
        model = CrossDIT(config)
    elif config.model_type == 'mmdit':
        model = MMDiT(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model
