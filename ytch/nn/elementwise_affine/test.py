import pytest
import torch

from ytch.correctness import assert_gradients_flow

from . import ElementwiseAffine

DIM = 10
SMALL_DIM = 3
BATCH_SIZE = 2


def test_elementwise_affine_defaults():
    layer = ElementwiseAffine(DIM)
    x = torch.randn(BATCH_SIZE, DIM)
    y = layer(x)

    assert y.shape == x.shape
    assert torch.allclose(layer.gamma, torch.ones(DIM))
    assert torch.allclose(layer.beta, torch.zeros(DIM))


def test_elementwise_affine_custom_init():
    layer = ElementwiseAffine(DIM, init_gamma=2.0, init_beta=0.5)
    assert torch.allclose(layer.gamma, torch.full((DIM,), 2.0))
    assert torch.allclose(layer.beta, torch.full((DIM,), 0.5))


def test_elementwise_affine_computation():
    layer = ElementwiseAffine(SMALL_DIM)
    layer.gamma.data = torch.tensor([2.0, 3.0, 4.0])
    layer.beta.data = torch.tensor([1.0, 2.0, 3.0])

    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = layer(x)
    assert torch.allclose(y, x * layer.gamma + layer.beta)


def test_elementwise_affine_multidim():
    layer = ElementwiseAffine(DIM)
    x = torch.randn(BATCH_SIZE, SMALL_DIM, DIM)
    assert layer(x).shape == x.shape


def test_elementwise_affine_dim_mismatch():
    layer = ElementwiseAffine(DIM)
    wrong_dim = DIM - 2
    with pytest.raises(AssertionError, match=f"Expected last dim {DIM}"):
        layer(torch.randn(BATCH_SIZE, wrong_dim))


def test_elementwise_affine_dtype_conversion():
    layer = ElementwiseAffine(DIM, dtype=torch.float32)
    x = torch.randn(SMALL_DIM, DIM, dtype=torch.float16)
    assert layer(x).dtype == torch.float16


def test_elementwise_affine_gradients():
    assert_gradients_flow(
        ElementwiseAffine(SMALL_DIM),
        torch.randn(BATCH_SIZE, SMALL_DIM, requires_grad=True),
    )
