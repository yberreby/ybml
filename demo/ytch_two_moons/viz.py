from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.text import Text
from mlflow import log_figure
from torch import Tensor, no_grad, tensor

from .config import VizConfig


def get_plot_filename(step: int) -> str:
    """Get standardized filename for decision surface plot."""
    return f"decision_surface_step_{step}.png"


def create_grid(
    x_np: np.ndarray[Any, np.dtype[np.floating[Any]]], cfg: VizConfig
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating[Any]]],
    np.ndarray[Any, np.dtype[np.floating[Any]]],
    Tensor,
]:
    """Create meshgrid for decision surface plotting (computed once)."""
    x_min = float(x_np[:, 0].min()) - cfg.plot_padding
    x_max = float(x_np[:, 0].max()) + cfg.plot_padding
    y_min = float(x_np[:, 1].min()) - cfg.plot_padding
    y_max = float(x_np[:, 1].max()) + cfg.plot_padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, cfg.grid_resolution),
        np.linspace(y_min, y_max, cfg.grid_resolution),
    )

    grid_points = tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    return xx, yy, grid_points


def create_plot_objects(
    x_np: np.ndarray[Any, np.dtype[np.floating[Any]]],
    y_np: np.ndarray[Any, np.dtype[np.integer[Any]]],
    xx: np.ndarray[Any, np.dtype[np.floating[Any]]],
    yy: np.ndarray[Any, np.dtype[np.floating[Any]]],
    cfg: VizConfig,
) -> tuple[Figure, AxesImage, Text]:
    """Create reusable plot objects (called once)."""
    fig, ax = plt.subplots(figsize=(cfg.figsize_width, cfg.figsize_height))
    assert isinstance(ax, Axes)

    extent = (float(xx.min()), float(xx.max()), float(yy.min()), float(yy.max()))
    dummy_z = np.zeros(xx.shape)
    im = ax.imshow(
        dummy_z,
        extent=extent,
        origin="lower",
        alpha=cfg.alpha_surface,
        cmap="RdYlBu",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    _ = ax.scatter(
        x_np[:, 0],
        x_np[:, 1],
        c=y_np,
        alpha=cfg.alpha_points,
        edgecolors="k",
        cmap="RdYlBu",
    )

    _ = ax.set_xlabel("Feature 1")
    _ = ax.set_ylabel("Feature 2")
    title = ax.set_title("")

    return fig, im, title


def update_decision_surface(
    model: nn.Module,
    grid_points: Tensor,
    xx: np.ndarray[Any, np.dtype[np.floating[Any]]],
    im: AxesImage,
    title: Text,
    step: int,
) -> Figure:
    """Update decision surface plot (fast, reuses objects)."""
    with no_grad():
        z = model(grid_points).argmax(dim=1).numpy()
    z = z.reshape(xx.shape)

    im.set_data(z)
    title.set_text(f"Decision Surface - Step {step}")

    fig = im.figure
    assert isinstance(fig, Figure)
    return fig


def log_decision_surface(
    model: nn.Module,
    grid_points: Tensor,
    xx: np.ndarray[Any, np.dtype[np.floating[Any]]],
    im: AxesImage,
    title: Text,
    step: int,
) -> None:
    """Update and log decision surface plot to MLflow."""
    fig = update_decision_surface(model, grid_points, xx, im, title, step)
    log_figure(fig, get_plot_filename(step))
