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
BLOCKS_PER_DIM = 4
IMAGE_SIZE = 128
N_STEPS = 500
BATCH_SIZE = 4096
LR = 1e-5
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
        layer.bias.data.zero_()


class ExpandingMLP(nn.Module):
    def __init__(self, use_norm, block_init, expand_init):
        super().__init__()

        layers = []

        # Input
        layers.append(RandomFourierFeaturesND(START_DIM // 2, device=DEVICE))
        layers.append(nn.Linear(START_DIM, START_DIM, device=DEVICE))
        layers.append(nn.ReLU())
        layers.append(
            nn.LayerNorm(START_DIM, device=DEVICE)
        )  # Always normalize after input

        # Hidden layers
        dim = START_DIM
        for _ in range(N_EXPANSIONS):
            # D→D blocks
            for _ in range(BLOCKS_PER_DIM):
                if use_norm:
                    layers.append(nn.LayerNorm(dim, device=DEVICE))
                layer = nn.Linear(dim, dim, device=DEVICE)
                if block_init == "ortho_block":
                    ortho_block_init_(layer)
                layers.append(layer)
                layers.append(nn.ReLU())

            # D→2D expansion
            if use_norm:
                layers.append(nn.LayerNorm(dim, device=DEVICE))
            layer = nn.Linear(dim, dim * 2, device=DEVICE)
            if expand_init == "ortho_expand":
                ortho_expand_init_(layer)
            layers.append(layer)
            layers.append(nn.ReLU())
            dim *= 2

        # Final D→D blocks
        for _ in range(BLOCKS_PER_DIM):
            if use_norm:
                layers.append(nn.LayerNorm(dim, device=DEVICE))
            layer = nn.Linear(dim, dim, device=DEVICE)
            if block_init == "ortho_block":
                ortho_block_init_(layer)
            layers.append(layer)
            layers.append(nn.ReLU())

        # Output
        final_dim = START_DIM * (2**N_EXPANSIONS)
        layers.append(
            nn.LayerNorm(final_dim, device=DEVICE)
        )  # Always normalize before output
        layers.append(nn.Linear(final_dim, 3, device=DEVICE))
        layers.append(nn.Tanh())
        layers.append(ElementwiseAffine(3, scale=5.0, device=DEVICE))

        self.model = nn.Sequential(*layers)

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


def main():
    coords, rgb, img_shape, img_np = load_image_coords(IMAGE_SIZE)

    def get_batch():
        idx = torch.randint(0, len(coords), (BATCH_SIZE,))
        return coords[idx], rgb[idx]

    # Create all variants
    variants = []
    for use_norm in [False, True]:
        for block_init in ["default", "ortho_block"]:
            for expand_init in ["default", "ortho_expand"]:
                name = f"{'norm' if use_norm else 'nonorm'}-{block_init}-{expand_init}"
                model = ExpandingMLP(use_norm, block_init, expand_init)
                print(f"Created {name}:")
                print(model)
                variants.append((name, model))

    print(f"Training {len(variants)} variants...")
    print(f"Params per model: {sum(p.numel() for p in variants[0][1].parameters()):,}")

    # Create trainers
    trainers = {
        name: Trainer(model, data=get_batch, lr=LR, n_steps=N_STEPS, disable_tqdm=i > 0)
        for i, (name, model) in enumerate(variants)
    }

    # Setup plots
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Loss curves (top row)
    ax_loss = fig.add_subplot(gs[0, :])
    loss_lines = {}
    for i, (name, _) in enumerate(variants):
        (loss_lines[name],) = ax_loss.plot([], [], label=name, alpha=0.7)
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=7, ncol=2)

    # Reconstructions (bottom 2 rows) - show 6 variants
    recon_axes = [fig.add_subplot(gs[1 + i // 3, i % 3]) for i in range(6)]
    recon_imgs = {}
    for i, (name, _) in enumerate(variants[:6]):
        recon_imgs[name] = recon_axes[i].imshow(np.zeros_like(img_np))
        recon_axes[i].set_title(name, fontsize=8)
        recon_axes[i].axis("off")

    # Track losses
    losses = {name: [] for name, _ in variants}

    # Training loop
    for step in range(N_STEPS):
        for name, trainer in trainers.items():
            metrics = next(trainer)
            losses[name].append(metrics["loss"].item())

        if step % UPDATE_INTERVAL == 0 or step == N_STEPS - 1:
            # Update loss curves
            for name in losses:
                loss_lines[name].set_data(range(len(losses[name])), losses[name])
            ax_loss.relim()
            ax_loss.autoscale_view()

            # Update reconstructions
            for name, model in variants[:6]:
                with torch.no_grad():
                    pred = model(coords, rgb)["pred"]
                    img = (
                        (pred.cpu() * IMAGENET_STD + IMAGENET_MEAN)
                        .clamp(0, 1)
                        .reshape(*img_shape, 3)
                        .numpy()
                    )
                    recon_imgs[name].set_data(img)

            fig.canvas.draw()
            fig.canvas.flush_events()

    plt.ioff()
    plt.show()

    print("\nFinal losses:")
    for name in losses:
        print(f"  {name:40s}: {losses[name][-1]:.6f}")


if __name__ == "__main__":
    main()
