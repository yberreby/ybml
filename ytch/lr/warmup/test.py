import pytest
from ytch.lr.warmup import get_warmup_steps_for_adam_beta2


def test_get_warmup_steps_for_adam_beta2():
    # Should crash for invalid beta2 values
    with pytest.raises(AssertionError):
        get_warmup_steps_for_adam_beta2(1.0)
    with pytest.raises(AssertionError):
        get_warmup_steps_for_adam_beta2(1.5)
    with pytest.raises(AssertionError):
        get_warmup_steps_for_adam_beta2(-0.1)

    # Edge case: beta2 = 0.0
    result_zero = get_warmup_steps_for_adam_beta2(0.0)
    assert result_zero == 2
    assert isinstance(result_zero, int)

    # Common case: torch default beta2 = 0.999
    result_default = get_warmup_steps_for_adam_beta2(0.999)
    assert result_default == 2000
    assert isinstance(result_default, int)
