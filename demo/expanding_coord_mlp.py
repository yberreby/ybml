from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ytch.constants import IMAGENET_MEAN, IMAGENET_STD
from ytch.device import get_sensible_device
from ytch.nn import ElementwiseAffine
from ytch.nn.rff import RandomFourierFeaturesND
from ytch.train import Trainer

# Hyperparams
START_DIM = 64
N_EXPANSIONS = 3
BLOCKS_PER_DIM = 2
IMAGE_SIZE = 128
N_STEPS = 2000
BATCH_SIZE = 4096
LR = 1e-4
UPDATE_INTERVAL = 10
DEVICE = get_sensible_device(forbid_mps=False)


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


class ExpandingMLP(nn.Module):
    def __init__(self, use_norm, block_init, expand_type):
        super().__init__()
        layers = []

        # Input
        layers.append(RandomFourierFeaturesND(START_DIM // 2, device=DEVICE))
        layers.append(nn.Linear(START_DIM, START_DIM, device=DEVICE))
        layers.append(nn.LayerNorm(START_DIM, elementwise_affine=False, device=DEVICE))
        layers.append(nn.ReLU())

        # Expanding trunk
        dim = START_DIM
        for _ in range(N_EXPANSIONS):
            layers.extend(self._build_d2d_blocks(dim, use_norm, block_init))
            layers.extend(self._build_expansion(dim, use_norm, expand_type))
            dim *= 2

        # Final blocks
        layers.extend(self._build_d2d_blocks(dim, use_norm, block_init))

        # Output head
        final_dim = START_DIM * (2**N_EXPANSIONS)
        layers.append(nn.LayerNorm(final_dim, elementwise_affine=False, device=DEVICE))
        layers.append(nn.Linear(final_dim, 3, device=DEVICE))
        layers.append(nn.Tanh())
        layers.append(ElementwiseAffine(3, scale=5.0, device=DEVICE))

        self.model = nn.Sequential(*layers)

    def _build_d2d_blocks(self, dim, use_norm, block_init):
        layers = []
        for _ in range(BLOCKS_PER_DIM):
            if use_norm:
                layers.append(
                    nn.LayerNorm(dim, elementwise_affine=False, device=DEVICE)
                )
            layer = nn.Linear(dim, dim, device=DEVICE)
            if block_init == "ortho_block":
                ortho_block_init_(layer)
            layers.append(layer)
            layers.append(nn.ReLU())
        return layers

    def _build_expansion(self, dim, use_norm, expand_type):
        layers = []
        if use_norm:
            layers.append(nn.LayerNorm(dim, elementwise_affine=False, device=DEVICE))

        if expand_type == "concat":
            layers.append(ConcatExpand())
        else:
            layer = nn.Linear(dim, dim * 2, device=DEVICE)
            if expand_type == "ortho_expand":
                ortho_expand_init_(layer)
            layers.append(layer)

        layers.append(nn.ReLU())
        return layers

    def forward(self, coords, rgb):
        pred = self.model(coords)
        return {"loss": F.mse_loss(pred, rgb), "pred": pred}


def load_image_coords(size):
    """Load test image and create coordinate dataset"""
    from PIL import Image

    img_path = Path(__file__).parent / "test_image.jpg"
    img = Image.open(img_path).resize((size, size))
    img_np = np.array(img).astype(np.float32) / 255.0

    H, W = img_np.shape[:2]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=DEVICE),
        torch.linspace(-1, 1, W, device=DEVICE),
        indexing="ij",
    )
    coords = torch.stack([x, y], dim=-1).reshape(-1, 2)
    rgb = torch.from_numpy(img_np).to(DEVICE).reshape(-1, 3)
    rgb_norm = (rgb - IMAGENET_MEAN.to(DEVICE)) / IMAGENET_STD.to(DEVICE)

    return coords, rgb_norm, (H, W), img_np


def create_variants():
    """Create all model variants to test"""
    variants = []
    for use_norm in [False, True]:
        for block_init in ["default", "ortho_block"]:
            for expand_type in ["default", "ortho_expand", "concat"]:
                if block_init == "default" and not use_norm:
                    print("This run is doomed, so skipping.")
                    continue
                name = f"{'norm' if use_norm else 'nonorm'}-{block_init}-{expand_type}"
                model = ExpandingMLP(use_norm, block_init, expand_type)
                variants.append((name, model))
    return variants


def setup_training_viz(variants, img_np):
    """Setup live training visualization"""
    plt.ion()
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)

    # Loss curves (row 0, left)
    ax_loss = fig.add_subplot(gs[0, :2])
    loss_lines = {
        name: ax_loss.plot([], [], label=name, alpha=0.7)[0] for name, _ in variants
    }
    ax_loss.set(xlabel="Step", ylabel="Loss", yscale="log")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=7, ncol=2)

    # Gradient norm curves (row 0, right)
    ax_gnorm = fig.add_subplot(gs[0, 2:])
    gnorm_lines = {
        name: ax_gnorm.plot([], [], label=name, alpha=0.7)[0] for name, _ in variants
    }
    ax_gnorm.set(xlabel="Step", ylabel="Grad Norm (pre-clip)", yscale="log")
    ax_gnorm.grid(True, alpha=0.3)
    ax_gnorm.legend(fontsize=7, ncol=2)

    # Gradient norm vs depth (row 1, full width)
    ax_depth = fig.add_subplot(gs[1, :])
    depth_lines = {
        name: ax_depth.plot([], [], label=name, alpha=0.7, marker="o", markersize=2)[0]
        for name, _ in variants
    }
    ax_depth.set(xlabel="Layer Depth", ylabel="Grad Norm / First Hidden", yscale="log")
    ax_depth.grid(True, alpha=0.3)
    ax_depth.legend(fontsize=7, ncol=2)

    # Reconstruction images (rows 2-4)
    n_variants = len(variants)
    recon_axes = [fig.add_subplot(gs[2 + i // 4, i % 4]) for i in range(n_variants)]
    recon_imgs = {}
    for i, (name, _) in enumerate(variants):
        recon_imgs[name] = recon_axes[i].imshow(np.zeros_like(img_np))
        recon_axes[i].set_title(name, fontsize=8)
        recon_axes[i].axis("off")

    return (
        fig,
        (ax_loss, ax_gnorm, ax_depth),
        (loss_lines, gnorm_lines, depth_lines),
        recon_imgs,
    )


def update_viz(
    axes,
    line_dicts,
    losses,
    gnorms,
    recon_imgs,
    variants,
    coords,
    rgb,
    img_shape,
    batch_size,
):
    """Update all visualization elements"""
    ax_loss, ax_gnorm, ax_depth = axes
    loss_lines, gnorm_lines, depth_lines = line_dicts

    # Update loss curves
    for name, line in loss_lines.items():
        line.set_data(range(len(losses[name])), [x.item() for x in losses[name]])
    ax_loss.relim()
    ax_loss.autoscale_view()

    # Update gradient norm curves
    for name, line in gnorm_lines.items():
        line.set_data(range(len(gnorms[name])), gnorms[name])
    ax_gnorm.relim()
    ax_gnorm.autoscale_view()

    # Update gradient norm vs depth
    for name, model in variants:
        gnorms_normalized = compute_layer_grad_norms(model, coords, rgb, batch_size)
        if gnorms_normalized:
            depth_lines[name].set_data(range(len(gnorms_normalized)), gnorms_normalized)
    ax_depth.relim()
    ax_depth.autoscale_view()

    # Update reconstructions
    for name, model in variants:
        with torch.no_grad():
            pred = model(coords, rgb)["pred"]
            img = (
                (pred.cpu() * IMAGENET_STD + IMAGENET_MEAN)
                .clamp(0, 1)
                .reshape(*img_shape, 3)
                .numpy()
            )
            recon_imgs[name].set_data(img)


def compute_layer_grad_norms(model, coords, rgb, batch_size):
    """Compute gradient norms for each Linear layer in model"""
    model.zero_grad()
    idx = torch.randint(0, len(coords), (batch_size,))
    pred = model.model(coords[idx])
    loss = F.mse_loss(pred, rgb[idx])
    loss.backward()

    layer_gnorms = []
    first_hidden_gnorm = None

    for i, layer in enumerate(model.model):
        if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
            gnorm = layer.weight.grad.norm().item()
            layer_gnorms.append(gnorm)
            if first_hidden_gnorm is None and i > 1:  # Skip RFF + input proj
                first_hidden_gnorm = gnorm

    # Normalize by first hidden layer
    if first_hidden_gnorm and first_hidden_gnorm > 0:
        return [g / first_hidden_gnorm for g in layer_gnorms]
    return []


def main():
    # Load data
    coords, rgb, img_shape, img_np = load_image_coords(IMAGE_SIZE)

    def get_batch():
        idx = torch.randint(0, len(coords), (BATCH_SIZE,))
        return coords[idx], rgb[idx]

    # Create models
    variants = create_variants()
    print(f"Training {len(variants)} variants")
    print(f"Params per model: {sum(p.numel() for p in variants[0][1].parameters()):,}")

    # Create trainers
    trainers = {
        name: Trainer(model, data=get_batch, lr=LR, n_steps=N_STEPS, disable_tqdm=i > 0)
        for i, (name, model) in enumerate(variants)
    }

    # Setup visualization
    fig, axes, line_dicts, recon_imgs = setup_training_viz(variants, img_np)

    # Training loop
    losses = {name: [] for name, _ in variants}
    gnorms = {name: [] for name, _ in variants}

    for step in range(N_STEPS):
        # Step all trainers
        for name, trainer in trainers.items():
            metrics = next(trainer)
            losses[name].append(metrics["loss"])
            gnorms[name].append(metrics["gnpre"])

        # Update visualization
        if step % UPDATE_INTERVAL == 0 or step == N_STEPS - 1:
            update_viz(
                axes,
                line_dicts,
                losses,
                gnorms,
                recon_imgs,
                variants,
                coords,
                rgb,
                img_shape,
                BATCH_SIZE,
            )
            fig.canvas.draw()
            fig.canvas.flush_events()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
