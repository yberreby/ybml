import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402

from ymlx.nn.swiglu import SwiGLU, SwiGLUResidualBlock, default_hidden  # noqa: E402


def test_swiglu_shapes():
    mlp = SwiGLU(in_dim=48, hidden=128, out_dim=32)
    assert mlp(mx.random.normal((5, 48))).shape == (5, 32)


def test_residual_block_shapes_and_default_hidden():
    blk = SwiGLUResidualBlock(dim=96)
    assert blk(mx.random.normal((3, 96))).shape == (3, 96)
    assert default_hidden(96) % 8 == 0
