import torch

from ytch.correctness import assert_gradients_flow

from . import RandomFourierFeaturesND


def test_rff_shape():
    rff = RandomFourierFeaturesND(dims_per_coord=32)
    x = torch.randn(10, 2)
    y = rff(x)
    assert y.shape == (10, 64)  # 2 * dims_per_coord (cos + sin)


def test_rff_gradients():
    assert_gradients_flow(
        RandomFourierFeaturesND(dims_per_coord=16),
        torch.randn(5, 2, requires_grad=True),
    )
