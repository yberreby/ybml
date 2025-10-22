from typing import NamedTuple

import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from ytch.metrics import compute_grad_norm
from zclip import ZClip


class TrainingStepOutput(NamedTuple):
    """Output from a single training step."""

    logits: Tensor
    loss: Tensor
    lr: float
    grad_norm_pre_clip: Tensor
    grad_norm_post_clip: Tensor


def training_step(
    batch,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    zclip: ZClip,
) -> TrainingStepOutput:
    """Execute single training step. Expects batch as (x, y) tuple."""
    optimizer.zero_grad()

    result = model(*batch)
    loss = result["loss"]
    logits = result["logits"]

    loss.backward()

    grad_norm_pre_clip = compute_grad_norm(model)
    _ = zclip.step(model)
    grad_norm_post_clip = compute_grad_norm(model)

    optimizer.step()
    scheduler.step()

    current_lrs = scheduler.get_last_lr()
    assert len(current_lrs) == 1
    current_lr = current_lrs[0]

    return TrainingStepOutput(
        logits=logits,
        loss=loss,
        lr=current_lr,
        grad_norm_pre_clip=grad_norm_pre_clip,
        grad_norm_post_clip=grad_norm_post_clip,
    )
