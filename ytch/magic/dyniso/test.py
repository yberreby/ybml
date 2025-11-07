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
    _ = torch.manual_seed(42)

    layers = [nn.Linear(dim, dim) for _ in range(depth)]
    for layer in layers:
        ortho_block_init_(layer)

    x = torch.randn(bs, dim)
    x_first_hidden = torch.relu(layers[1](x))

    for layer in layers:
        x = torch.relu(layer(x))

    # Activation preservation
    act_ratio = x.std() / x_first_hidden.std()
    assert act_ratio > 0.9, f"Activations collapsed: {act_ratio:.3f}"

    # Gradient preservation
    loss = x.pow(2).mean()
    _ = loss.backward()

    # DEBUG:
    # for layer in layers:
    #     print("Grad norm:", layer.weight.grad.norm())

    # SKIP THE INPUT LAYER!
    g_first_hidden = layers[1].weight.grad
    assert g_first_hidden is not None
    g_first_hidden = g_first_hidden.norm()

    g_middle = []
    for i in range(1, depth - 1):
        g = layers[i].weight.grad
        assert g is not None
        g_middle.append(g)
    g_middle = torch.stack([g.norm() for g in g_middle])

    # Gradients should flow (not vanish)
    assert g_first_hidden > 0.5, (
        f"First hidden layer grad vanished: {g_first_hidden:.6f}"
    )

    # Gradients should not explode/vanish relative to first hidden layer
    grad_ratio = g_middle.mean() / g_first_hidden
    assert grad_ratio < 1.2, f"Gradients exploded with depth, ratio: {grad_ratio:.1f}"
    assert grad_ratio > 0.8, f"Gradients vanished depth, ratio: {grad_ratio:.1f}"
