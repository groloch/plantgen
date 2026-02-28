from dataclasses import dataclass


@dataclass
class MLflowConfig:
    experiment_name: str
    tracking_uri: str