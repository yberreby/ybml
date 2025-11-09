from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from PIL import Image

from ytch.constants import IMAGENET_MEAN, IMAGENET_STD
from ytch.device import get_sensible_device
from ytch.magic.dyniso import ortho_block_init_
from ytch.nn import ElementwiseAffine
from ytch.nn.rff import RandomFourierFeaturesND
from ytch.train import Trainer


@dataclass
class Config:
    image_size: int = 128
    io_dim: int = 128
    hidden_dim: int = 128
    n_blocks: int = 8
    n_steps: int = 500
    lr: float = 1e-4
    batch_size: int = 4096
    update_interval: int = 10
    dpi: int = 150
    save_mp4: bool = True
    fps: int = 30
    device: str | torch.device = get_sensible_device(forbid_mps=False)

    def __post_init__(self):
        assert self.hidden_dim % 2 == 0, "hidden_dim must be even for Fourier features"


InitScheme = Literal["default", "dyniso"]


@dataclass
class ArchConfig:
    use_skip: bool
    use_prenorm: bool

    @property
    def name(self) -> str:
        skip = "Skip" if self.use_skip else "NoSkip"
        norm = "PreNorm" if self.use_prenorm else "NoNorm"
        return f"{skip}-{norm}"


MODELS = [
    # Sensible baseline: default init, prenorm skip
    ("default", ArchConfig(use_skip=True, use_prenorm=True)),
    # Naive skip - also pretty bad, at high depths
    ("default", ArchConfig(use_skip=True, use_prenorm=False)),
    # The way it's meant to be used
    ("dyniso", ArchConfig(use_skip=False, use_prenorm=False)),
    ("dyniso", ArchConfig(use_skip=False, use_prenorm=True)),
    ("dyniso", ArchConfig(use_skip=True, use_prenorm=True)),
]


def make_all_models(cfg: Config) -> dict[str, nn.Module]:
    return {f"{init}-{arch.name}": make_model(cfg, arch, init) for init, arch in MODELS}


def load_image(path: str, size: int) -> np.ndarray:
    img = Image.open(path).resize((size, size))
    return np.array(img).astype(np.float32) / 255.0


def create_coordinate_dataset(img_np, device):
    H, W = img_np.shape[:2]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    coords = torch.stack([x, y], dim=-1).reshape(-1, 2)
    rgb = torch.from_numpy(img_np).to(device).reshape(-1, 3)
    return coords, (rgb - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device), (H, W)


class SingleActivationMLPBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        init_scheme: InitScheme,
        use_prenorm: bool,
        device=None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=device)

        match init_scheme:
            case "dyniso":
                ortho_block_init_(self.fc1)
                ortho_block_init_(self.fc2)
            case "default":
                pass

        if use_prenorm:
            self.maybe_norm = nn.LayerNorm(hidden_dim, device=device)
        else:
            self.maybe_norm = nn.Identity()

    def forward(self, x):
        x = self.maybe_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CoordMLP(nn.Module):
    def __init__(
        self,
        io_dim: int,
        hidden_dim: int,
        n_blocks: int,
        use_skip: bool = False,
        use_prenorm: bool = False,
        init_scheme: InitScheme = "default",
        device=None,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.use_prenorm = use_prenorm

        self.act = nn.ReLU()

        self.input_map = nn.Sequential(
            RandomFourierFeaturesND(io_dim // 2, device=device),
            nn.Linear(io_dim, hidden_dim, device=device),
        )
        self.output_map = nn.Sequential(
            nn.Linear(hidden_dim, io_dim, device=device),
            nn.LayerNorm(io_dim, device=device),
            nn.Linear(io_dim, 3, device=device),
            nn.Tanh(),
            # Bounded but rescalable output range
            ElementwiseAffine(3, init_gamma=5.0, device=device),
        )

        # Hidden blocks: the bulk
        self.hidden_blocks = nn.ModuleList(
            [
                SingleActivationMLPBlock(
                    hidden_dim, init_scheme, use_prenorm=self.use_prenorm, device=device
                )
                for _ in range(n_blocks)
            ]
        )
        self.hidden_lns = (
            nn.ModuleList(
                [nn.LayerNorm(hidden_dim, device=device) for _ in range(n_blocks)]
            )
            if use_prenorm
            else None
        )

    def forward(self, coords):
        x = self.input_map(coords)

        # Bulk: blocks
        for i, block in enumerate(self.hidden_blocks):
            block_input = self.hidden_lns[i](x) if self.hidden_lns else x
            residual = block(block_input)
            if self.use_skip:
                # Just sum.
                x = x + residual
            else:
                # MUST apply activation if not using skip,
                # otherwise we end up with a bunch of Linear -> Linear
                x = self.act(residual)

        x = self.output_map(x)
        return x


def make_model(cfg: Config, arch: ArchConfig, init: InitScheme) -> CoordMLP:
    return CoordMLP(
        io_dim=cfg.io_dim,
        hidden_dim=cfg.hidden_dim,
        n_blocks=cfg.n_blocks,
        use_skip=arch.use_skip,
        use_prenorm=arch.use_prenorm,
        init_scheme=init,
        device=torch.device(cfg.device),
    )


class ModelWrapper(nn.Module):
    """Wraps CoordMLP to return dict with 'loss' for Trainer API."""

    def __init__(self, model: CoordMLP):
        super().__init__()
        self.model = model

    def forward(self, coords: torch.Tensor, rgb: torch.Tensor) -> dict:
        pred = self.model(coords)
        return {"loss": F.mse_loss(pred, rgb), "pred": pred}


class Runner:
    def __init__(self, cfg: Config, img_np: np.ndarray):
        self.cfg = cfg
        self.img_np = img_np
        self.coords, self.rgb, self.img_shape = create_coordinate_dataset(
            img_np, device=cfg.device
        )
        self.models = make_all_models(cfg)

    def setup_live_plots(self):
        plt.ion()
        names = list(self.models.keys())

        # Metrics figure (separate, not fullscreen)
        self.metrics_fig, self.metric_axes = plt.subplots(1, 3, figsize=(15, 4))
        self.lines = {metric: {} for metric in ["loss", "grad", "grad_by_depth"]}

        # Metric configurations: (key, xlabel, ylabel, title, plot_kwargs)
        metric_specs = [
            ("loss", "Step", "Loss", "Training Loss", {}),
            ("grad", "Step", "Grad Norm", "Gradient Norm", {}),
            (
                "grad_by_depth",
                "Block",
                "Grad Norm",
                "Grad by Depth",
                {"marker": "o", "ms": 3},
            ),
        ]

        for i, name in enumerate(names):
            color = f"C{i}"
            linestyle = "-" if "Skip-" in name else "--"
            for j, spec in enumerate(metric_specs):
                key, _, _, _, plot_kwargs = spec
                (self.lines[key][name],) = self.metric_axes[j].plot(
                    [],
                    [],
                    label=name,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.8,
                    **plot_kwargs,
                )

        for j, spec in enumerate(metric_specs):
            _, xlabel, ylabel, title, _ = spec
            self.metric_axes[j].set(
                xlabel=xlabel, ylabel=ylabel, title=title, yscale="log"
            )
            self.metric_axes[j].grid(True, alpha=0.3)

        self.metric_axes[0].legend(loc="upper right", fontsize=8)
        self.metrics_fig.tight_layout()

        # Reconstructions figure (2x3 grid for video)
        n_total = len(names) + 1  # GT + models
        assert n_total == 6, f"Expected 6 images (1 GT + 5 models), got {n_total}"
        self.recon_fig, recon_axes = plt.subplots(2, 3, figsize=(12, 8))
        recon_axes = recon_axes.flatten()

        self.recon_imgs = {}
        # Ground truth
        recon_axes[0].imshow(self.img_np)
        recon_axes[0].set_title("Ground Truth", fontsize=10)
        recon_axes[0].axis("off")

        # Model reconstructions
        for i, name in enumerate(names, start=1):
            self.recon_imgs[name] = recon_axes[i].imshow(np.zeros_like(self.img_np))
            recon_axes[i].set_title(name, fontsize=10)
            recon_axes[i].axis("off")

        self.recon_fig.tight_layout()

    def update_plots(self, metrics):
        # Update metrics
        for name, data in metrics.items():
            n = len(data["loss"])
            losses = [x.detach().cpu().item() for x in data["loss"]]
            grad_norms = [x.detach().cpu().item() for x in data["grad_norm"]]
            grad_by_depth = [x.detach().cpu().item() for x in data["grad_by_depth"][-1]]
            self.lines["loss"][name].set_data(range(n), losses)
            self.lines["grad"][name].set_data(range(n), grad_norms)
            self.lines["grad_by_depth"][name].set_data(
                range(len(grad_by_depth)), grad_by_depth
            )

        for ax in self.metric_axes:
            ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))
            ax.relim()
            ax.autoscale_view()

        if self.metric_axes[2].get_ylim()[0] <= 0:
            self.metric_axes[2].set_ylim(bottom=1e-10)

        # Update reconstructions
        preds = {}
        for name, model in self.models.items():
            with torch.no_grad():
                pred = model(self.coords)
                preds[name] = pred

        for name, pred in preds.items():
            img = self._denormalize_rgb(pred)
            self.recon_imgs[name].set_data(img)

    def _denormalize_rgb(self, pred):
        return (
            (pred.cpu() * IMAGENET_STD + IMAGENET_MEAN)
            .clamp(0, 1)
            .reshape(*self.img_shape, 3)
            .numpy()
        )

    def train(self):
        # Wrap models for Trainer API
        wrapped = {name: ModelWrapper(m) for name, m in self.models.items()}

        def get_batch():
            idx = torch.randint(0, len(self.coords), (self.cfg.batch_size,))
            return self.coords[idx], self.rgb[idx]

        # Create trainers (all but first have tqdm disabled)
        trainers = {}
        for i, (name, model) in enumerate(wrapped.items()):
            trainers[name] = Trainer(
                model,
                data=get_batch,
                lr=self.cfg.lr,
                n_steps=self.cfg.n_steps,
                disable_tqdm=i > 0,
            )

        self.metrics = defaultdict(
            lambda: {"loss": [], "grad_norm": [], "grad_by_depth": []}
        )
        self.setup_live_plots()

        writer = None
        if self.cfg.save_mp4:
            from matplotlib.animation import FFMpegWriter

            writer = FFMpegWriter(fps=self.cfg.fps)
            mp4_path = Path(__file__).parent / "training.mp4"
            writer.setup(self.recon_fig, str(mp4_path), dpi=self.cfg.dpi)

        # Step all trainers synchronously
        for step in range(self.cfg.n_steps):
            for name, trainer in trainers.items():
                step_metrics = next(trainer)

                # Compute extra metrics for comparison
                grad_by_depth = [
                    torch.stack(
                        [
                            p.grad.norm().detach()
                            for p in b.parameters()
                            if p.grad is not None
                        ]
                    ).sum()
                    for b in self.models[name].hidden_blocks
                ]

                m = self.metrics[name]
                m["loss"].append(step_metrics["loss"].detach())
                m["grad_norm"].append(torch.tensor(step_metrics["gnpre"]))
                m["grad_by_depth"].append(grad_by_depth)

            if step % self.cfg.update_interval == 0 or step == self.cfg.n_steps - 1:
                self.update_plots(self.metrics)
                for fig in [self.metrics_fig, self.recon_fig]:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                if writer:
                    writer.grab_frame()

        if writer:
            writer.finish()

        plt.ioff()

    def save_results(self):
        output_path = Path(__file__).parent / "reconstructions.png"
        self.recon_fig.savefig(output_path, dpi=self.cfg.dpi, bbox_inches="tight")
        print(f"\nSaved reconstructions PNG to {output_path}")
        if self.cfg.save_mp4:
            print(f"Saved MP4 to {Path(__file__).parent / 'training.mp4'}")

    def print_stats(self):
        print("\n=== Final Stats ===")
        for name in self.models.keys():
            print(
                f"{name:30s}: loss={self.metrics[name]['loss'][-1]:.3e}, grad={self.metrics[name]['grad_norm'][-1]:.3e}"
            )


def main(cfg: Config):
    img_path = Path(__file__).parent / "test_image.jpg"
    assert img_path.exists(), f"Error: {img_path} not found"

    print(f"Loading {img_path}")
    img_np = load_image(str(img_path), cfg.image_size)

    print(
        f"Config: dim={cfg.hidden_dim}, blocks={cfg.n_blocks}, lr={cfg.lr}, steps={cfg.n_steps}"
    )

    runner = Runner(cfg, img_np)
    print(f"Training {len(runner.models)} variants...")
    for name, model in runner.models.items():
        print(f"{name} has {sum(x.numel() for x in model.parameters())} parameters")

    runner.train()
    runner.save_results()
    runner.print_stats()


if __name__ == "__main__":
    tyro.cli(main)
