import math


def get_warmup_steps_for_adam_beta2(beta2: float) -> int:
    # "On the adequacy of untuned warmup for adaptive optimization"
    # Jerry Ma, Denis Yarats
    # https://arxiv.org/abs/1910.04209
    assert 0 <= beta2 < 1, f"beta2 must be in [0, 1), got {beta2=}"
    return math.ceil(2 / (1 - beta2))


def get_linear_scaled_lr(
    base_lr: float, batch_size: int, base_batch_size: int = 1
) -> float:
    """Scale learning rate linearly with batch size.

    Linear scaling rule: lr = base_lr * (batch_size / base_batch_size)
    """
    assert batch_size > 0, f"batch_size must be positive, got {batch_size=}"
    assert base_batch_size > 0, (
        f"base_batch_size must be positive, got {base_batch_size=}"
    )
    return base_lr * (batch_size / base_batch_size)
