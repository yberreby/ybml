import math


def get_warmup_steps_for_adam_beta2(beta2: float) -> int:
    # "On the adequacy of untuned warmup for adaptive optimization"
    # Jerry Ma, Denis Yarats
    # https://arxiv.org/abs/1910.04209
    assert 0 <= beta2 < 1
    return math.ceil(2 / (1 - beta2))
