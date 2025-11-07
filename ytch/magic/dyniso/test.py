import pytest
import torch
import torch.nn as nn

from ytch.magic.dyniso import ortho_block_init_


def test_requires_even_dimensions():
    """Odd dimensions must be rejected."""
    layer = nn.Linear(3, 4)
    with pytest.raises(AssertionError, match="even"):
        ortho_block_init_(layer)


def test_dynamical_isometry():
    """Activations and gradients should not collapse through deep network."""
    depth, dim, bs = 128, 64, 512

    layers = [nn.Linear(dim, dim) for _ in range(depth)]
    for layer in layers:
        ortho_block_init_(layer)

    x = torch.randn(bs, dim)
    x_first = torch.relu(layers[0](x))

    for layer in layers:
        x = torch.relu(layer(x))

    # Activation preservation
    act_ratio = x.std() / x_first.std()
    assert act_ratio > 1e-2, f"Activations collapsed: {act_ratio:.3f}"

    # Gradient preservation
    loss = x.pow(2).mean()
    _ = loss.backward()

    # DEBUG:
    # for layer in layers:
    #     print("Grad norm:", layer.weight.grad.norm())

    first_grad = layers[0].weight.grad
    assert first_grad is not None
    g_first = first_grad.norm()

    g_middle = []
    for i in range(1, depth - 1):
        g = layers[i].weight.grad
        assert g is not None
        g_middle.append(g)
    g_middle = torch.stack([g.norm() for g in g_middle])
    grad_ratio = g_middle.mean() / g_first
    assert grad_ratio < 100, f"Gradients exploding: {grad_ratio:.1f}"
