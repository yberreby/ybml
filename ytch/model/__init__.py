import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters (trainable and non-trainable)
    """
    return sum(p.numel() for p in model.parameters())
