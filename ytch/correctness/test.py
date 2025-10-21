import pytest
import torch
from ytch.correctness import assert_shape


def test_assert_shape_passes():
    x = torch.zeros(2, 3, 4)
    assert_shape(x, (2, 3, 4))


def test_assert_shape_fails():
    x = torch.zeros(2, 3, 4)
    with pytest.raises(
        AssertionError, match="expected shape \\(2, 3, 5\\), got \\(2, 3, 4\\)"
    ):
        assert_shape(x, (2, 3, 5))
