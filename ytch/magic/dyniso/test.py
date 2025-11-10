import pytest
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

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


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_device():
    """Smoketest: ortho_block_init_ works on MPS device."""
    layer = nn.Linear(4, 4, device="mps")
    ortho_block_init_(layer)
    assert layer.weight.device.type == "mps"


def test_plot_isometry():
    """Visual verification: plot activation and gradient flow vs default init."""
    depth, dim, bs = 128, 64, 512

    def run_experiment(init_fn):
        _ = torch.manual_seed(42)
        layers = [nn.Linear(dim, dim) for _ in range(depth)]
        for layer in layers:
            init_fn(layer)

        x = torch.randn(bs, dim)

        # Forward - skip first (projection) layer in tracking
        x = torch.relu(layers[0](x))  # Project but don't track
        act_stds = []
        for layer in layers[1:]:
            x = torch.relu(layer(x))
            act_stds.append(x.std().item())

        # Backward - skip first layer in tracking
        loss = x.pow(2).mean()
        loss.backward()
        grad_norms = []
        for layer in layers[1:]:
            assert layer.weight.grad is not None
            grad_norms.append(layer.weight.grad.norm().item())

        return act_stds, grad_norms

    def noop(_):
        pass

    experiments = [
        ("ortho_block_init_", ortho_block_init_, "C0"),
        ("default init", noop, "C1"),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for i, (title, init_fn, color) in enumerate(experiments):
        _, grad_norms = run_experiment(init_fn)
        axs[i].plot(grad_norms, linewidth=2, color=color)
        axs[i].set_xlabel("Hidden layer depth")
        axs[i].set_ylabel("Gradient norm")
        axs[i].set_title(title)
        axs[i].set_yscale("log")
        axs[i].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "isometry.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")
