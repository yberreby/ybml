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


@hydra.main(version_base=None, config_name="two_moons")
def main(cfg: DictConfig) -> None:
    _ = manual_seed(cfg.dataset.seed)
    np.random.seed(cfg.dataset.seed)

    _ = set_experiment(cfg.logging.experiment_name)

    with start_run():
        _ = log_params(
            {
                "dataset_noise_std": cfg.dataset.noise_std,
                "batch_size": cfg.training.batch_size,
                "n_steps": cfg.training.n_steps,
                "base_lr": cfg.training.base_lr,
                "hidden_dim": cfg.model.hidden_dim,
            }
        )

        # Create grid for visualization using a sample dataset
        x_grid, y_grid = make_moons(
            n_samples=cfg.dataset.grid_samples,
            noise=cfg.dataset.noise_std,
            random_state=cfg.dataset.seed,
        )
        assert isinstance(x_grid, np.ndarray)
        assert isinstance(y_grid, np.ndarray)
        xx, yy, grid_points = create_grid(x_grid, cfg.viz)
        fig, im, title = create_plot_objects(x_grid, y_grid, xx, yy, cfg.viz)

        model = TwoMoonsClassifier(cfg.model.hidden_dim)
        n_params = count_parameters(model)
        print(f"Model parameters: {n_params:,}")
        _ = log_param("n_parameters", n_params)

        # Print gradient norms at initialization
        print("\nGradient norms at initialization:")
        x_init, y_init = make_moons(
            n_samples=cfg.training.batch_size, noise=cfg.dataset.noise_std
        )
        x_init = tensor(x_init, dtype=torch.float32)
        y_init = tensor(y_init, dtype=torch.long)
        result_init = model(x_init, y_init)
        result_init["loss"].backward()
        print_grad_norms(model, prefix="  ")
        model.zero_grad()
        print()

        optimizer = AdamW(model.parameters(), lr=cfg.training.base_lr)
        scheduler = get_linear_warmup_scheduler(optimizer)
        zclip = ZClip()

        # Log initial decision surface before any training
        log_decision_surface(model.mlp, grid_points, xx, im, title, 0)

        assert cfg.training.n_steps > 0, "n_steps must be positive"

        step_output = None
        acc = 0.0

        for step in trange(cfg.training.n_steps, desc="Training"):
            # Generate fresh batch at each step (infinite data regime)
            x_batch, y_batch = make_moons(
                n_samples=cfg.training.batch_size, noise=cfg.dataset.noise_std
            )
            x_batch = tensor(x_batch, dtype=torch.float32)
            y_batch = tensor(y_batch, dtype=torch.long)

            batch = (x_batch, y_batch)
            step_output = training_step(batch, model, optimizer, scheduler, zclip)

            acc = compute_accuracy(step_output.logits, y_batch)

            _ = log_metrics(
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
                step + 1
            ) % cfg.logging.plot_every_n_steps == 0 or step == cfg.training.n_steps - 1:
                log_decision_surface(model.mlp, grid_points, xx, im, title, step + 1)

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
        print("\nMLflow run completed! View results with: `uv run mlflow ui`")


if __name__ == "__main__":
    main()
