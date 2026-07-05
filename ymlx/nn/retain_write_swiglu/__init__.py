"""Retain/write-gated SwiGLU cell: one artifact for depth stacks AND recurrence in time.

`RetainWriteSwiGLUCell` — decoupled retain/write gates over a leaky carry,
with a SwiGLU candidate — maps (state, input_encoding) -> state through a
content-gated carry and an independently gated SwiGLU write:

    h  = [rms_norm(x); rms_norm(e)]
    a  = exp(-8 * softplus(nu) * sigmoid(W_r h))    per-channel retain, ~0.91 at init
    s  = sigmoid(W_i h)                             independent write gate
    x' = a * x + s * SwiGLU(h)                      SwiGLU output-proj zero-init

`RetainWriteSwiGLUStep` chains two distinct cells: one *time step* when
weight-tied across applications, or one depth stage when stacked untied.
Decode at step boundaries. Supply the current input encoding every call; a
held input is the special case where `e` does not change.

The cell owns all input scaling (RMSNorms on state and encoding): any
encoder/decoder plugs in without calibration. At exact init the write path
is silent (zero-init output projection) and the map is x -> a*x; the write
wakes on the first optimizer steps.

Empirical basis (single seed, d=128, coordinate-regression harness,
2026-07-04, repo 2026-07-03-fun-exps @ ef32d27): vs an identically tuned
SwiGLU baseline, ~+3 dB on static depth (weight-tied, 24 steps deep) and
~+3 dB on changing-input recurrence with a near-seamless mid-rollout target
switch, at +25% params / +28% FLOPs. Weight-tied, it also matched the best
fully-untied model measured in that harness (a different gated variant,
separate run) at 28x fewer params. Design decisions and refuted alternatives: that repo's
RESEARCH_LOG.md.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from ymlx.nn.swiglu import SwiGLU, default_hidden

_GATE_SCALE = 8.0
_RETAIN_BIAS = -2.0
_WRITE_BIAS = 2.0
_NU_INIT = math.log(math.expm1(0.1))  # softplus(nu) = 0.1


class RetainWriteSwiGLUCell(nn.Module):
    def __init__(self, dim: int, e_dim: int, hidden: int | None = None):
        super().__init__()
        h_dim = dim + e_dim
        self.norm_x = nn.RMSNorm(dim)
        self.norm_e = nn.RMSNorm(e_dim)
        self.w_r = nn.Linear(h_dim, dim)
        self.w_r.bias = mx.full((dim,), _RETAIN_BIAS)
        self.w_i = nn.Linear(h_dim, dim)
        self.w_i.bias = mx.full((dim,), _WRITE_BIAS)
        self.nu = mx.full((dim,), _NU_INIT)
        self.mlp = SwiGLU(h_dim, hidden or default_hidden(dim), dim)
        self.mlp.w_o.weight = mx.zeros_like(self.mlp.w_o.weight)

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        h = mx.concatenate([self.norm_x(x), self.norm_e(e)], axis=-1)
        a = mx.exp(-_GATE_SCALE * nn.softplus(self.nu) * mx.sigmoid(self.w_r(h)))
        s = mx.sigmoid(self.w_i(h))
        return a * x + s * self.mlp(h)


class RetainWriteSwiGLUStep(nn.Module):
    """Two distinct cells = one time step (tied) or one depth stage (untied)."""

    def __init__(self, dim: int, e_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.e_dim = e_dim or dim // 2
        self.cell_a = RetainWriteSwiGLUCell(dim, self.e_dim)
        self.cell_b = RetainWriteSwiGLUCell(dim, self.e_dim)

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        return self.cell_b(self.cell_a(x, e), e)

    def initial_state(self, batch_shape: tuple[int, ...]) -> mx.array:
        return mx.zeros((*batch_shape, self.dim))
