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


def print_grad_norms(model: nn.Module, prefix: str = "") -> None:
    """Print L2 gradient norm for each parameter in the model.

    Args:
        model: Model whose parameter gradient norms to print
        prefix: Optional prefix for the output lines
    """
    prefix_str = f"{prefix} " if prefix else ""
    overall_norm = compute_grad_norm(model).item()
    print(f"{prefix_str}Overall: {overall_norm:.6f}")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{prefix_str}{name}: {grad_norm:.6f}")
        else:
            print(f"{prefix_str}{name}: None")
