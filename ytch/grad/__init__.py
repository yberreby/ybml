import torch
import torch.nn as nn


def compute_grad_norm(model: nn.Module) -> float:
    """Compute the L2 norm of gradients across all model parameters.

    Uses torch.nn.utils.clip_grad_norm_ with infinite threshold to compute
    the norm without actually clipping.

    Args:
        model: Model whose gradient norm to compute

    Returns:
        L2 norm of all gradients
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
