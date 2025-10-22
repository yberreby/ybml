from typing import NamedTuple

import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from ytch.grad import compute_grad_norm
from zclip import ZClip


class TrainingStepOutput(NamedTuple):
    """Output from a single training step."""

    logits: Tensor
    loss: Tensor
    lr: float
    grad_norm_pre_clip: Tensor
    grad_norm_post_clip: Tensor


def training_step(
    model: nn.Module,
    batch_x: Tensor,
    batch_y: Tensor,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    criterion: nn.Module,
    zclip: ZClip,
) -> TrainingStepOutput:
    """Execute single training step."""
    optimizer.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
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
