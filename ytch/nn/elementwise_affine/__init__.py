import torch
import torch.nn as nn
from torch import Tensor


class ElementwiseAffine(nn.Module):
    """
    Learnable per-dimension affine: y = scale * x + bias

    Reparameterized so learnable components start at zero:
      scale = init_scale + delta_scale (delta_scale init to 0)
      bias = init_bias + delta_bias (delta_bias init to 0)

    This ensures AdamW weight decay biases towards identity transform.

    Args:
        dim: Feature dimension
        scale: None (init to 1) | float/Tensor (init value)
        bias: None/True (init to 0) | float/Tensor (init value) | False (disabled)
    """

    init_scale: Tensor
    init_bias: Tensor | None

    def __init__(
        self,
        dim: int,
        scale: float | Tensor | None = None,
        bias: float | Tensor | bool | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        scale_val = 1.0 if scale is None else scale
        if isinstance(scale_val, (int, float)):
            init_s = torch.full((dim,), float(scale_val), device=device, dtype=dtype)
        else:
            init_s = scale_val.to(device=device, dtype=dtype)
        self.register_buffer("init_scale", init_s)
        self.delta_scale = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

        if bias is False:
            self.register_buffer("init_bias", None)
            self.delta_bias = None
        else:
            bias_val = 0.0 if bias is None or bias is True else bias
            if isinstance(bias_val, (int, float)):
                init_b = torch.full((dim,), float(bias_val), device=device, dtype=dtype)
            else:
                init_b = bias_val.to(device=device, dtype=dtype)
            self.register_buffer("init_bias", init_b)
            self.delta_bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    @property
    def scale(self) -> Tensor:
        return self.init_scale + self.delta_scale

    @property
    def bias(self) -> Tensor | None:
        if self.init_bias is None:
            return None
        assert self.delta_bias is not None
        return self.init_bias + self.delta_bias

    def forward(self, x: Tensor) -> Tensor:
        scale = self.scale
        assert x.size(-1) == scale.numel(), (
            f"Expected last dim {scale.numel()}, got {x.size(-1)}"
        )
        shape = (1,) * (x.ndim - 1) + (-1,)
        out = x * scale.view(shape).to(x.dtype)
        bias = self.bias
        if bias is not None:
            out = out + bias.view(shape).to(x.dtype)
        return out
