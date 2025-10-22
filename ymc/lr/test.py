import pytest
from ymc.lr import get_warmup_steps_for_adam_beta2, get_linear_scaled_lr

BETA2_DEFAULT = 0.999
BETA2_CUSTOM = 0.99


def test_get_warmup_steps_for_adam_beta2():
    assert get_warmup_steps_for_adam_beta2(BETA2_DEFAULT) == 2000
    assert get_warmup_steps_for_adam_beta2(BETA2_CUSTOM) == 200


def test_get_warmup_steps_for_adam_beta2_invalid():
    with pytest.raises(AssertionError, match="beta2 must be in"):
        get_warmup_steps_for_adam_beta2(1.5)


def test_get_linear_scaled_lr():
    assert get_linear_scaled_lr(1e-3, 64, 32) == pytest.approx(2e-3)
    assert get_linear_scaled_lr(1e-3, 32, 32) == pytest.approx(1e-3)
    assert get_linear_scaled_lr(1e-3, 16, 32) == pytest.approx(5e-4)
