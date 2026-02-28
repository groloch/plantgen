from dataclasses import dataclass


@dataclass
class PlantNetDataConfig:
    image_size: int
    num_classes: int
    batch_size: int
    num_workers: int
    augmentation_mode: str


@dataclass
class PlantNetTTIDataConfig:
    image_size: int
    num_classes: int
    batch_size: int
    num_workers: int
    annotations_path: str
    similarity_threshold: float
    precomputed_latents: bool
    latents_path: str = None
    latent_dim: int = None
    latent_size: int = None