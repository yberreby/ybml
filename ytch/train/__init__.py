import time
from collections.abc import Callable, Iterator
from typing import Any

import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from zclip import ZClip

from ymc.lr import get_warmup_steps_for_adam_beta2
from ytch.lr.warmup import get_linear_warmup_scheduler
from ytch.metrics import compute_grad_norm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        data: Iterator[Any] | Callable[[], Any],
        lr: float,
        weight_decay: float = 1e-4,
        n_steps: int = 5000,
        normalize_grads: bool = True,
        beta1: float = 0.8,  # default
        beta2: float = 0.95,  # assuming phase transitions happen fast enough that 20 steps is a good enough horizon
        on_step: Callable[[nn.Module, int, tuple[Any, ...], dict[str, Any]], None]
        | None = None,
        log_interval_seconds: float = 0.1,
        disable_tqdm: bool = False,
    ):
        self.model = model
        self.n_steps = n_steps
        self.log_interval_seconds = log_interval_seconds
        self.on_step = on_step
        self.normalize_grads = normalize_grads

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if not disable_tqdm:
            print(
                f"Using {self.optimizer.__class__.__name__}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                f"weight_decay={self.optimizer.param_groups[0]['weight_decay']:.2e}"
            )

        # Warmup + cosine decay scheduler
        _beta1, beta2 = self.optimizer.param_groups[0].get(
            "betas", self.optimizer.defaults["betas"]
        )
        warmup_steps = get_warmup_steps_for_adam_beta2(beta2)
        warmup_steps = min(warmup_steps, n_steps // 2)
        print(f"Using warmup_steps={warmup_steps} out of {n_steps}")

        warmup_sched = get_linear_warmup_scheduler(self.optimizer, warmup_steps)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=self.n_steps, eta_min=0)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )

        self.zclip = ZClip(warmup_steps=0)
        self.get_batch = data if callable(data) else lambda: next(data)

        self.start_time = time.time()
        self.last_update = self.start_time
        self.total_samples = 0
        self.step_count = 0
        self.metrics = {}
        self.pbar = tqdm(total=n_steps, unit="opt", disable=disable_tqdm)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        if self.step_count >= self.n_steps:
            self.pbar.close()
            raise StopIteration

        batch = self.get_batch()
        batch_size = len(batch[0]) if isinstance(batch, tuple) else len(batch)
        self.total_samples += batch_size

        self.optimizer.zero_grad()
        out = self.model(*batch) if isinstance(batch, tuple) else self.model(batch)
        out["loss"].backward()

        gnpre = self.zclip.step(self.model)

        if self.normalize_grads:
            mean_grad_norm = self.zclip.mean or gnpre
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 / (mean_grad_norm + 1e-8))

        been_a_while = time.time() - self.last_update >= self.log_interval_seconds
        should_refresh = (
            been_a_while or self.step_count == 0 or self.step_count == self.n_steps - 1
        )
        gnpost = compute_grad_norm(self.model) if should_refresh else None

        self.optimizer.step()
        self.scheduler.step()

        if self.on_step is not None:
            self.on_step(self.model, self.step_count, batch, out)

        elapsed = time.time() - self.start_time
        self.metrics = {
            "samples/s": self.total_samples / elapsed if elapsed > 0 else 0,
            "loss": out["loss"],
            "lr": self.scheduler.get_last_lr()[0],
            "gnpre": gnpre,
        }

        if should_refresh:
            self.metrics["gnpost"] = gnpost
            display = {
                k: f"{v.item() if isinstance(v, Tensor) else v:.3e}"
                if k == "lr"
                else v.item()
                if isinstance(v, Tensor)
                else v
                for k, v in self.metrics.items()
            }
            self.pbar.set_postfix(display)
            self.last_update = time.time()

        self.pbar.update(1)
        self.step_count += 1
        return self.metrics


def train(
    model: nn.Module,
    data: Iterator[Any] | Callable[[], Any],
    lr: float,
    **kwargs,
) -> dict[str, Any]:
    trainer = Trainer(model, data, lr, **kwargs)
    metrics = {}
    for metrics in trainer:
        pass
    return metrics
