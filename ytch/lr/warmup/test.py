import pytest
import torch.nn as nn
from torch.optim import Adam, AdamW
from ytch.lr.warmup import get_warmup_steps_for_adam_beta2, get_linear_warmup_scheduler


def test_get_warmup_steps_for_adam_beta2():
    with pytest.raises(AssertionError):
        get_warmup_steps_for_adam_beta2(1.0)
    with pytest.raises(AssertionError):
        get_warmup_steps_for_adam_beta2(-0.1)

    assert get_warmup_steps_for_adam_beta2(0.0) == 2
    assert get_warmup_steps_for_adam_beta2(0.999) == 2000


@pytest.mark.parametrize("optimizer_class", [Adam, AdamW])
def test_get_linear_warmup_scheduler(optimizer_class):
    model = nn.Linear(10, 10)
    max_lr = 1e-3
    optimizer = optimizer_class(model.parameters(), lr=max_lr)
    scheduler = get_linear_warmup_scheduler(optimizer)
    warmup_steps = 2000

    assert scheduler.get_last_lr()[0] == pytest.approx(max_lr / warmup_steps)
    for _ in range(warmup_steps - 1):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(max_lr)


def test_get_linear_warmup_scheduler_different_beta2():
    model = nn.Linear(10, 10)
    lr = 1e-3
    weight_beta2, bias_beta2 = 0.999, 0.99
    optimizer = Adam(
        [
            {"params": model.weight, "lr": lr, "betas": (0.9, weight_beta2)},
            {"params": model.bias, "lr": lr, "betas": (0.9, bias_beta2)},
        ]
    )
    scheduler = get_linear_warmup_scheduler(optimizer)

    weight_warmup = get_warmup_steps_for_adam_beta2(weight_beta2)
    bias_warmup = get_warmup_steps_for_adam_beta2(bias_beta2)
    assert scheduler.get_last_lr() == pytest.approx(
        [lr / weight_warmup, lr / bias_warmup]
    )

    for _ in range(max(weight_warmup, bias_warmup) - 1):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([lr, lr])
