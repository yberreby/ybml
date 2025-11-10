from typing import Any

import torch
from torch import Tensor


class GradMultiply(torch.autograd.Function):
    """Forward: identity. Backward: scale gradients by a factor."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, scale: float | Tensor) -> Tensor:
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Tensor, None]:
        (grad,) = grad_outputs
        return grad * ctx.scale, None


def grad_multiply(x: Tensor, scale: float | Tensor) -> Tensor:
    """Scale gradients by factor (0=detach, 1=identity)."""
    result = GradMultiply.apply(x, scale)
    assert isinstance(result, Tensor)
    return result
