import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402

import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from ymlx.nn.retain_write_swiglu import (  # noqa: E402
    RetainWriteSwiGLUCell,
    RetainWriteSwiGLUStep,
)


def wake(step: RetainWriteSwiGLUStep, scale: float = 0.01) -> RetainWriteSwiGLUStep:
    """Give the write projections a small random value, as early training does.

    At exact init (zero write, zero state) a rollout from initial_state is
    identically zero and only the write projection receives gradient.
    """
    for cell in (step.cell_a, step.cell_b):
        cell.mlp.w_o.weight = scale * mx.random.normal(cell.mlp.w_o.weight.shape)
    return step


def test_shapes():
    step = RetainWriteSwiGLUStep(dim=64)
    x = step.initial_state((7,))
    out = step(x, mx.random.normal((7, step.e_dim)))
    assert out.shape == (7, 64)


def test_pure_decay_at_init():
    # zero-init write => the cell is x -> a*x exactly
    cell = RetainWriteSwiGLUCell(dim=64, e_dim=32)
    x = mx.random.normal((5, 64))
    e = mx.random.normal((5, 32))
    out = cell(x, e)
    ratio = out / x
    assert (ratio > 0).all().item() and (ratio < 1).all().item()


def test_held_input_rollout_bounded():
    step = wake(RetainWriteSwiGLUStep(dim=64))
    e = mx.random.normal((9, step.e_dim))
    x = step.initial_state((9,))
    for _ in range(128):
        x = step(x, e)
    norm = mx.linalg.norm(x, axis=-1).mean().item()
    assert 1e-3 < norm < 1e3


def test_gradients_reach_all_components_post_wake():
    step = wake(RetainWriteSwiGLUStep(dim=64))
    e = mx.random.normal((4, step.e_dim))
    target = mx.random.normal((4, 64))

    def loss(m: RetainWriteSwiGLUStep) -> mx.array:
        x = m.initial_state((4,))
        for _ in range(3):
            x = m(x, e)
        return mx.mean((x - target) ** 2)

    grads = nn.value_and_grad(step, loss)(step)[1]
    for key, grad in tree_flatten(grads):
        assert isinstance(grad, mx.array)
        assert mx.linalg.norm(grad).item() > 0, key
