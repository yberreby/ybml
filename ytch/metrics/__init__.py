import torch
import torch.nn as nn
from torch import Tensor


def compute_grad_norm(model: nn.Module) -> Tensor:
    """Compute the L2 norm of gradients across all model parameters.

    Uses torch.nn.utils.clip_grad_norm_ with infinite threshold to compute
    the norm without actually clipping.

    Args:
        model: Model whose gradient norm to compute

    Returns:
        L2 norm of all gradients as a 0-d tensor
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
