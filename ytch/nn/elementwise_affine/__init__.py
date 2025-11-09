import torch
import torch.nn as nn
from torch import Tensor


class ElementwiseAffine(nn.Module):
    """Learnable per-dimension affine along the last axis: y = x * gamma + beta."""

    def __init__(
        self,
        dim: int,
        init_gamma: float | None = None,
        init_beta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.dim = dim

        def make_param(init_val, default_fn):
            if init_val is None:
                return default_fn(dim, device=device, dtype=dtype)
            return torch.full(
                size=(dim,), fill_value=init_val, device=device, dtype=dtype
            )

        self.gamma = nn.Parameter(make_param(init_gamma, torch.ones))
        self.beta = nn.Parameter(make_param(init_beta, torch.zeros))

    def forward(self, x: Tensor) -> Tensor:
        assert x.size(-1) == self.gamma.numel(), (
            f"Expected last dim {self.gamma.numel()}, got {x.size(-1)}"
        )
        # Broadcast gamma/beta: (1, 1, ..., 1, dim) for arbitrary rank
        shape = (1,) * (x.ndim - 1) + (-1,)
        gamma = self.gamma.view(shape).to(x.dtype)
        beta = self.beta.view(shape).to(x.dtype)
        return x * gamma + beta

    def __repr__(self) -> str:
        return f"ElementwiseAffine(dim={self.dim})"
