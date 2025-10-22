import matplotlib.pyplot as plt
import numpy as np
import torch
from mlflow import log_metrics, log_param, log_params, set_experiment, start_run
from sklearn.datasets import make_moons
from torch import manual_seed, tensor
from torch.optim import AdamW
from tqdm import trange
from ytch.lr.warmup import get_linear_warmup_scheduler
from ytch.model import count_parameters
from zclip import ZClip

from .model import TwoMoonsClassifier
from .config import (
    ACC_PRECISION,
    BASE_LR,
    BATCH_SIZE,
    DATASET_NOISE_STD,
    EXPERIMENT_NAME,
    GRID_SAMPLES,
    HIDDEN_DIM,
    LOG_PLOT_EVERY_N_STEPS,
    LOSS_PRECISION,
    LR_PRECISION,
    N_STEPS,
    SEED,
)
from .metrics import compute_accuracy
from .train import training_step
from .viz import create_grid, create_plot_objects, log_decision_surface


def main():
    _ = manual_seed(SEED)
    np.random.seed(SEED)

    _ = set_experiment(EXPERIMENT_NAME)

    with start_run():
        _ = log_params(
            {
                "dataset_noise_std": DATASET_NOISE_STD,
                "batch_size": BATCH_SIZE,
                "n_steps": N_STEPS,
                "base_lr": BASE_LR,
                "hidden_dim": HIDDEN_DIM,
            }
        )

        # Create grid for visualization using a sample dataset
        x_grid, y_grid = make_moons(
            n_samples=GRID_SAMPLES, noise=DATASET_NOISE_STD, random_state=SEED
        )
        assert isinstance(x_grid, np.ndarray)
        assert isinstance(y_grid, np.ndarray)
        xx, yy, grid_points = create_grid(x_grid)
        fig, im, title = create_plot_objects(x_grid, y_grid, xx, yy)

        model = TwoMoonsClassifier(HIDDEN_DIM)
        n_params = count_parameters(model)
        print(f"Model parameters: {n_params:,}")
        _ = log_param("n_parameters", n_params)

        optimizer = AdamW(model.parameters(), lr=BASE_LR)
        scheduler = get_linear_warmup_scheduler(optimizer)
        zclip = ZClip()

        # Log initial decision surface before any training
        log_decision_surface(model.mlp, grid_points, xx, im, title, 0)

        assert N_STEPS > 0, "N_STEPS must be positive"

        step_output = None
        acc = 0.0

        for step in trange(N_STEPS, desc="Training"):
            # Generate fresh batch at each step (infinite data regime)
            x_batch, y_batch = make_moons(n_samples=BATCH_SIZE, noise=DATASET_NOISE_STD)
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

            if (step + 1) % LOG_PLOT_EVERY_N_STEPS == 0 or step == N_STEPS - 1:
                log_decision_surface(model.mlp, grid_points, xx, im, title, step + 1)

        plt.close(fig)

        if N_STEPS > 0:
            assert step_output is not None
            print(
                f"Final | Loss: {step_output.loss.item():.{LOSS_PRECISION}f} | "
                + f"Acc: {acc:.{ACC_PRECISION}f} | "
                + f"LR: {step_output.lr:.{LR_PRECISION}f} | "
                + f"Grad Norm Pre: {step_output.grad_norm_pre_clip.item():.{LOSS_PRECISION}f} | "
                + f"Grad Norm Post: {step_output.grad_norm_post_clip.item():.{LOSS_PRECISION}f}"
            )
        print("\nMLflow run completed! View results with: `uv run mlflow ui`")


if __name__ == "__main__":
    main()
