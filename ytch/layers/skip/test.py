import torch
import torch.nn as nn
from ytch.correctness import assert_gradients_flow
from ytch.layers.skip import Skip

BATCH_SIZE = 2
DIM = 10


def test_skip_basic():
    block = nn.Linear(DIM, DIM)
    layer = Skip(block)
    x = torch.randn(BATCH_SIZE, DIM)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x + block(x))


def test_skip_gradients():
    assert_gradients_flow(
        Skip(nn.Linear(DIM, DIM)), torch.randn(BATCH_SIZE, DIM, requires_grad=True)
    )
