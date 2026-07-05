import mlx.core as mx
import mlx.nn as nn


class SwiGLU(nn.Module):
    """The SwiGLU MLP: in_dim -> hidden (silu-gated product) -> out_dim.

    The bare candidate, no norm, no residual — compose it yourself.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.w_uv = nn.Linear(in_dim, 2 * hidden, bias=False)
        self.w_o = nn.Linear(hidden, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        p, q = mx.split(self.w_uv(x), 2, axis=-1)
        return self.w_o(nn.silu(p) * q)


def default_hidden(dim: int) -> int:
    """The Llama-style SwiGLU expansion width: 8*dim/3, rounded to 8."""
    return round(8 * dim / 3 / 8) * 8


class SwiGLUResidualBlock(nn.Module):
    """Prenorm residual SwiGLU sublayer, as in modern Transformer MLPs."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden or default_hidden(dim), dim)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.mlp(self.norm(x))
