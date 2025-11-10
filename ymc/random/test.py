import numpy as np
import pytest
from ymc.random import sample_by_tail_ratio


def test_edge_cases():
    assert np.all(sample_by_tail_ratio(5, 10, 0.0, size=100) == 5)
    assert np.all(sample_by_tail_ratio(5, 5, 0.5, size=50) == 5)

    samples = sample_by_tail_ratio(0, 5, 0.5, size=(10, 20))
    assert samples.shape == (10, 20)
    assert np.all((samples >= 0) & (samples <= 5))


def test_tail_ratio_property():
    """P(max)/P(min) ≈ tail_ratio."""
    a, b, tail_ratio, n = 0, 10, 0.3, 50000
    samples = sample_by_tail_ratio(a, b, tail_ratio, size=n)
    counts = np.bincount(samples - a, minlength=b - a + 1)
    ratio = (counts[-1] / n) / (counts[0] / n)
    assert pytest.approx(ratio, rel=0.2) == tail_ratio


def test_invalid_inputs():
    with pytest.raises(AssertionError, match="tail_ratio must be in"):
        sample_by_tail_ratio(0, 10, -0.1)
    with pytest.raises(AssertionError, match="a must be <= b"):
        sample_by_tail_ratio(10, 5, 0.5)
