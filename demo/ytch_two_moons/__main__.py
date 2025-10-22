import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from mlflow import log_metrics, log_param, log_params, set_experiment, start_run
from omegaconf import DictConfig
from sklearn.datasets import make_moons
from torch import manual_seed, tensor
from torch.optim import AdamW
from tqdm import trange
from ytch.lr.warmup import get_linear_warmup_scheduler
from ytch.metrics import print_grad_norms
from ytch.model import count_parameters
from zclip import ZClip

from .config import TwoMoonsConfig
from .metrics import compute_accuracy
from .model import TwoMoonsClassifier
from .train import training_step
from .viz import create_grid, create_plot_objects, log_decision_surface

cs = ConfigStore.instance()
cs.store(name="two_moons", node=TwoMoonsConfig)


def generate_batch(n_samples: int, noise_std: float, mean: np.ndarray, std: np.ndarray):
    """Generate and standardize a batch of two moons data."""
    x, y = make_moons(n_samples=n_samples, noise=noise_std)
    x = (x - mean) / std
    return tensor(x, dtype=torch.float32), tensor(y, dtype=torch.long)


def print_init_grad_norms(
    model, batch_size: int, noise_std: float, mean: np.ndarray, std: np.ndarray
) -> None:
    """Print gradient norms at initialization."""
    print("\nGradient norms at initialization:")
    x_init, y_init = generate_batch(batch_size, noise_std, mean, std)
    result_init = model(x_init, y_init)
    result_init["loss"].backward()
    print_grad_norms(model, prefix="  ")
    model.zero_grad()
    print()


def run_training(
    cfg: TwoMoonsConfig, log_mlflow: bool = True, enable_viz: bool = True
) -> float:
    """Run training with given config. Returns final accuracy."""
    _ = manual_seed(cfg.dataset.seed)
    np.random.seed(cfg.dataset.seed)

    # Compute input statistics for standardization
    x_stats, _ = make_moons(
        n_samples=10000, noise=cfg.dataset.noise_std, random_state=cfg.dataset.seed
    )
    assert isinstance(x_stats, np.ndarray)
    input_mean = x_stats.mean(axis=0)
    input_std = x_stats.std(axis=0)

    if log_mlflow:
        _ = set_experiment(cfg.logging.experiment_name)

    def maybe_log_params(params):
        if log_mlflow:
            _ = log_params(params)

    def maybe_log_metrics(metrics, step):
        if log_mlflow:
            _ = log_metrics(metrics, step=step)

    def maybe_log_param(key, value):
        if log_mlflow:
            _ = log_param(key, value)

    maybe_log_params(
        {
            "dataset_noise_std": cfg.dataset.noise_std,
            "batch_size": cfg.training.batch_size,
            "n_steps": cfg.training.n_steps,
            "base_lr": cfg.training.base_lr,
            "weight_decay": cfg.training.weight_decay,
            "hidden_dim": cfg.model.hidden_dim,
            "smart_output_init": cfg.model.smart_output_init,
        }
    )

    # Create grid for visualization using a sample dataset
    if enable_viz:
        x_grid, y_grid = make_moons(
            n_samples=cfg.dataset.grid_samples,
            noise=cfg.dataset.noise_std,
            random_state=cfg.dataset.seed,
        )
        assert isinstance(x_grid, np.ndarray)
        assert isinstance(y_grid, np.ndarray)
        x_grid = (x_grid - input_mean) / input_std
        xx, yy, grid_points = create_grid(x_grid, cfg.viz)
        fig, im, title = create_plot_objects(x_grid, y_grid, xx, yy, cfg.viz)
    else:
        fig = im = title = xx = yy = grid_points = None  # type: ignore[assignment]

    model = TwoMoonsClassifier(
        cfg.model.hidden_dim, smart_output_init=cfg.model.smart_output_init
    )
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    maybe_log_param("n_parameters", n_params)

    print_init_grad_norms(
        model, cfg.training.batch_size, cfg.dataset.noise_std, input_mean, input_std
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.base_lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = get_linear_warmup_scheduler(optimizer)
    zclip = ZClip()

    # Log initial decision surface before any training
    if log_mlflow and enable_viz:
        assert (
            grid_points is not None
            and xx is not None
            and im is not None
            and title is not None
        )
        log_decision_surface(model.mlp, grid_points, xx, im, title, 0)

    assert cfg.training.n_steps > 0, "n_steps must be positive"

    step_output = None
    acc = 0.0

    for step in trange(cfg.training.n_steps, desc="Training"):
        # Generate fresh batch at each step (infinite data regime)
        x_batch, y_batch = generate_batch(
            cfg.training.batch_size, cfg.dataset.noise_std, input_mean, input_std
        )
        batch = (x_batch, y_batch)
        step_output = training_step(batch, model, optimizer, scheduler, zclip)

        acc = compute_accuracy(step_output.logits, y_batch)

        maybe_log_metrics(
            {
                "loss": step_output.loss.item(),
                "accuracy": acc,
                "lr": step_output.lr,
                "grad_norm_pre_clip": step_output.grad_norm_pre_clip.item(),
                "grad_norm_post_clip": step_output.grad_norm_post_clip.item(),
            },
            step=step,
        )

        if (
            log_mlflow
            and enable_viz
            and (
                (step + 1) % cfg.logging.plot_every_n_steps == 0
                or step == cfg.training.n_steps - 1
            )
        ):
            assert (
                grid_points is not None
                and xx is not None
                and im is not None
                and title is not None
            )
            log_decision_surface(model.mlp, grid_points, xx, im, title, step + 1)

    if enable_viz:
        plt.close(fig)

    if cfg.training.n_steps > 0:
        assert step_output is not None
        print(
            f"Final | Loss: {step_output.loss.item():.{cfg.formatting.loss_precision}f} | "
            + f"Acc: {acc:.{cfg.formatting.acc_precision}f} | "
            + f"LR: {step_output.lr:.{cfg.formatting.lr_precision}f} | "
            + f"Grad Norm Pre: {step_output.grad_norm_pre_clip.item():.{cfg.formatting.loss_precision}f} | "
            + f"Grad Norm Post: {step_output.grad_norm_post_clip.item():.{cfg.formatting.loss_precision}f}"
        )
    if log_mlflow:
        print("\nMLflow run completed! View results with: `uv run mlflow ui`")

    return acc


@hydra.main(version_base=None, config_name="two_moons")
def main(cfg: DictConfig) -> None:
    """Entry point with Hydra config."""
    from typing import cast

    from omegaconf import OmegaConf

    with start_run():
        _ = run_training(
            cast(TwoMoonsConfig, OmegaConf.to_object(cfg)), log_mlflow=True
        )


if __name__ == "__main__":
    main()
