from dataclasses import dataclass


@dataclass
class TrainingConfig:
    learning_rate: float
    weight_decay: float
    num_epochs: int
    grad_clip: float
    logdir: str
    save_every: int
    log_every: int
    profile: bool

@dataclass
class VAETrainingConfig(TrainingConfig):
    iaf: bool
    ckpt_path: str = None


@dataclass
class ClassifierTrainingConfig(TrainingConfig):
    pass


@dataclass
class FlowMatchingTrainingConfig(TrainingConfig):
    vae_ckpt_path: str
    text_encoder: str
    warmup_steps: int
    gradient_accumulation_steps: int
    timestep_distribution: str    


__all__ = [
    'TrainingConfig',
    'VAETrainingConfig',
    'ClassifierTrainingConfig',
    'FlowMatchingTrainingConfig'
]
