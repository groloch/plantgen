import random
import torch


def model_parameters(model: torch.nn.Module) -> int:
    """Calculate the total and trainable parameters of a model.

    Args:
        model (torch.nn.Module): The model to calculate parameters for.

    Returns:
        tuple: A tuple containing total parameters and trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_parameters

def denormalize(tensor: torch.Tensor, means = [0.485, 0.456, 0.406], stds = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    Denormalize a tensor from ImageNet normalization back to [0, 1] range.

    Args:
        tensor: Tensor with ImageNet normalization applied (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    Returns:
        Tensor in [0, 1] range suitable for image display
    """
    mean = torch.tensor(means).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(stds).view(1, 3, 1, 1).to(tensor.device)

    denormalized = tensor * std + mean

    return torch.clamp(denormalized, 0, 1)
