import time
from typing import Callable, Iterator

import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from tqdm import trange
from zclip import ZClip

from ytch.lr.warmup import get_linear_warmup_scheduler
from ytch.metrics import compute_grad_norm


def train(
    model: nn.Module,
    data: Iterator | Callable,
    lr: float,
    weight_decay: float = 1e-4,
    n_steps: int = 5000,
    bar_refresh_interval_seconds: float = 0.1,
    on_step: Callable[[nn.Module, int, tuple, dict], None] | None = None,
) -> dict:
    """Train with Adam + linear warmup + ZClip + tqdm.

    Args:
        on_step: Optional callback(model, step, batch, outputs) called every step.
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(
        f"Using {optimizer.__class__.__name__}, lr={optimizer.param_groups[0]['lr']:.2e}, weight_decay={optimizer.param_groups[0]['weight_decay']:.2e}"
    )
    scheduler = get_linear_warmup_scheduler(optimizer)
    zclip = ZClip(max_grad_norm=None)
    get_batch = data if callable(data) else lambda: next(data)

    start_time = time.time()
    last_update = start_time
    total_samples = 0
    metrics = {}

    pbar = trange(n_steps, unit="opt")  # opt steps
    for step in pbar:
        batch = get_batch()
        batch_size = len(batch[0]) if isinstance(batch, tuple) else len(batch)
        total_samples += batch_size

        optimizer.zero_grad()

        # Forward
        out = model(*batch) if isinstance(batch, tuple) else model(batch)
        # Backward
        out["loss"].backward()
        # Adaptive gradient clipping
        gnpre = zclip.step(model)

        # Optional metrics computation
        been_a_while = time.time() - last_update >= bar_refresh_interval_seconds
        should_refresh = been_a_while or step == 0 or step == n_steps - 1
        gnpost = None
        if should_refresh:
            gnpost = compute_grad_norm(model)

        # Optimizer and LR scheduler steps
        _ = optimizer.step()
        scheduler.step()

        # Callbacks
        if on_step is not None:
            on_step(model, step, batch, out)

        # Progress bar update if applicable
        elapsed_since_start = time.time() - start_time
        samples_per_sec = (
            total_samples / elapsed_since_start if elapsed_since_start > 0 else 0
        )

        metrics = {
            "samples/s": samples_per_sec,
            "loss": out["loss"],
            "lr": scheduler.get_last_lr()[0],
            "gnpre": gnpre,
        }

        # Infrequent due to e.g. GPU syncs from .item()
        if should_refresh:
            assert gnpost is not None
            metrics.update({"gnpost": gnpost})
            display = {}
            for k, v in metrics.items():
                val = v.item() if isinstance(v, Tensor) else v
                display[k] = f"{val:.3e}" if k == "lr" else val
            pbar.set_postfix(display)
            last_update = time.time()

    return metrics
