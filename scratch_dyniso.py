"""Testing dynamical isometry for expanding architectures."""

import torch
import torch.nn as nn
import numpy as np


def ortho_block_init_(layer):
    """2x2 block orthogonal for D→D"""
    assert layer.in_features == layer.out_features and layer.out_features % 2 == 0
    h = layer.out_features // 2
    with torch.no_grad():
        w0 = torch.empty(h, h)
        torch.nn.init.orthogonal_(w0)
        layer.weight.data[:h, :h] = w0
        layer.weight.data[:h, h:] = -w0
        layer.weight.data[h:, :h] = -w0
        layer.weight.data[h:, h:] = w0
        if layer.bias is not None:
            layer.bias.data.zero_()


def ortho_expand_init_(layer):
    """[[W0], [-W0]] for D→2D"""
    assert layer.out_features == 2 * layer.in_features
    d = layer.in_features
    with torch.no_grad():
        w0 = torch.empty(d, d)
        torch.nn.init.orthogonal_(w0)
        layer.weight.data[:d, :] = w0
        layer.weight.data[d:, :] = -w0
        if layer.bias is not None:
            layer.bias.data.zero_()


class ConcatExpand(nn.Module):
    """Parameter-free expansion: [x; -x]"""
    def forward(self, x):
        return torch.cat([x, -x], dim=-1)


def compute_jacobian_spectral_norm(model, input_dim):
    """Compute spectral norm of input-output Jacobian."""
    x = torch.randn(1, input_dim, requires_grad=True)
    J = torch.autograd.functional.jacobian(model, x).squeeze()
    S = torch.linalg.svdvals(J)
    return S[0].item()


def measure_over_seeds(model_fn, input_dim, n_seeds):
    """Measure spectral norm over multiple random seeds."""
    specs = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model = model_fn()
        spec = compute_jacobian_spectral_norm(model, input_dim)
        specs.append(spec)
    return np.mean(specs), np.std(specs)


def build_mixed(start_d, n_expansions, blocks_per_dim, expand_type, relu_after_expand):
    """
    Build mixed architecture: [D→D blocks] → [D→2D expand] → repeat.

    D→D blocks always use ortho_block_init.
    Expansion type can be: 'default', 'ortho_expand', or 'concat'.
    relu_after_expand: whether to add ReLU after expansion layer.
    """
    layers = []
    dim = start_d

    for _ in range(n_expansions):
        # D→D blocks with ortho_block init
        for _ in range(blocks_per_dim):
            block = nn.Linear(dim, dim)
            ortho_block_init_(block)
            layers.append(block)
            layers.append(nn.ReLU())

        # D→2D expansion (varies)
        if expand_type == 'concat':
            layers.append(ConcatExpand())
        else:
            expand = nn.Linear(dim, dim*2)
            if expand_type == 'ortho_expand':
                ortho_expand_init_(expand)
            layers.append(expand)

        if relu_after_expand:
            layers.append(nn.ReLU())
        dim *= 2

    # Final D→D blocks
    for _ in range(blocks_per_dim):
        block = nn.Linear(dim, dim)
        ortho_block_init_(block)
        layers.append(block)
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


# Config
START_D = 16
N_SEEDS = 10

expansion_types = [
    ('default', 'Default Linear init'),
    ('ortho_expand', 'Ortho [[W0], [-W0]]'),
    ('concat', 'Concat [x; -x]'),
]

for blocks_per_dim in [0, 4]:
    for relu_after_expand in [True, False]:
        arch_name = "Pure expansion" if blocks_per_dim == 0 else "Mixed [D→D blocks] → [D→2D expand]"
        relu_desc = "with ReLU" if relu_after_expand else "NO ReLU"

        print("=" * 70)
        print(f"{arch_name} ({relu_desc} after expansion)")
        print(f"blocks_per_dim={blocks_per_dim}, start_d={START_D}")
        print("=" * 70)
        print()

        for expand_type, desc in expansion_types:
            print(f"=== Expansion: {desc} ===")
            print()

            for n_exp in [1, 2, 3]:
                model = build_mixed(START_D, n_exp, blocks_per_dim, expand_type, relu_after_expand)

                if n_exp == 1 and relu_after_expand:
                    print(f"Example model ({n_exp} expansion):")
                    print(model)
                    print()

                mean, std = measure_over_seeds(
                    lambda e=expand_type, n=n_exp, b=blocks_per_dim, r=relu_after_expand:
                        build_mixed(START_D, n, b, e, r),
                    START_D,
                    N_SEEDS
                )

                total_layers = n_exp * (blocks_per_dim + 1) + blocks_per_dim
                print(f"  {n_exp} expansion(s) ({total_layers:2d} layers): {mean:.6f} ± {std:.6f}")

            print()

        print(f"Reference: √2 = {np.sqrt(2):.6f}")
        print()
        print()
