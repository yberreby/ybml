import torch
import torch.nn as nn
from torch import Tensor


class ElementwiseAffine(nn.Module):
    """
    Learnable per-dimension affine: y = scale * x + bias

    Args:
        dim: Feature dimension
        scale: None (init to 1) | float/Tensor (init value)
        bias: None/True (init to 0) | float/Tensor (init value) | False (disabled)

    Examples:
        ElementwiseAffine(512)                    # scale=1, bias=0
        ElementwiseAffine(512, scale=2.0)         # scale=2, bias=0
        ElementwiseAffine(512, bias=False)        # scale=1, no bias
        ElementwiseAffine(512, scale=0.01)        # scale=0.01, bias=0
    """

    def __init__(
        self,
        dim: int,
        scale: float | Tensor | None = None,
        bias: float | Tensor | bool | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        def _make_scale(val):
            if val is None:
                return nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
            if isinstance(val, (int, float)):
                return nn.Parameter(
                    torch.full((dim,), float(val), device=device, dtype=dtype)
                )
            assert isinstance(val, Tensor)
            return nn.Parameter(val.to(device=device, dtype=dtype))

        def _make_bias(val):
            if val is None or val is True:
                return nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
            if val is False:
                return None
            if isinstance(val, (int, float)):
                return nn.Parameter(
                    torch.full((dim,), float(val), device=device, dtype=dtype)
                )
            assert isinstance(val, Tensor)
            return nn.Parameter(val.to(device=device, dtype=dtype))

        self.scale = _make_scale(scale)
        self.bias = _make_bias(bias)

    def forward(self, x: Tensor) -> Tensor:
        assert x.size(-1) == self.scale.numel(), (
            f"Expected last dim {self.scale.numel()}, got {x.size(-1)}"
        )
        shape = (1,) * (x.ndim - 1) + (-1,)
        out = x * self.scale.view(shape).to(x.dtype)
        if self.bias is not None:
            out = out + self.bias.view(shape).to(x.dtype)
        return out
