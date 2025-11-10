import torch

from ytch.correctness import assert_gradients_flow

from . import LayerScale

DIM = 10
SMALL_DIM = 3
BATCH_SIZE = 2


def test_layerscale_defaults():
    layer = LayerScale(DIM)
    assert layer.scale is not None
    assert torch.allclose(layer.scale, torch.full((DIM,), 1e-6))
    assert layer.bias is None


def test_layerscale_custom_init():
    init_val = 1e-4
    layer = LayerScale(DIM, init_values=init_val)
    assert layer.scale is not None
    assert torch.allclose(layer.scale, torch.full((DIM,), init_val))


def test_layerscale_computation():
    layer = LayerScale(SMALL_DIM, init_values=0.5)
    x = torch.randn(BATCH_SIZE, SMALL_DIM)
    y = layer(x)
    assert torch.allclose(y, x * 0.5)


def test_layerscale_dtype_conversion():
    layer = LayerScale(DIM, dtype=torch.float32)
    x = torch.randn(SMALL_DIM, DIM, dtype=torch.float16)
    assert layer(x).dtype == torch.float16


def test_layerscale_gradients():
    assert_gradients_flow(
        LayerScale(SMALL_DIM),
        torch.randn(BATCH_SIZE, SMALL_DIM, requires_grad=True),
    )
