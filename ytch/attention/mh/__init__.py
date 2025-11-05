from torch import Tensor

from ytch import assert_shape


def to_multihead(x: Tensor, num_heads: int) -> Tensor:
    """Reshape to multi-head format: [B, N, D] -> [B, H, N, head_dim]."""
    B, N, D = x.shape
    head_dim = D // num_heads
    assert D % num_heads == 0, f"D={D} must be divisible by num_heads={num_heads}"
    x = x.view(B, N, num_heads, head_dim).transpose(1, 2)
    assert_shape(x, (B, num_heads, N, head_dim))
    return x


def from_multihead(x: Tensor) -> Tensor:
    """Reshape from multi-head format: [B, H, N, head_dim] -> [B, N, D]."""
    B, H, N, head_dim = x.shape
    D = H * head_dim
    x = x.transpose(1, 2).reshape(B, N, D)
    assert_shape(x, (B, N, D))
    return x
