import pytest
import torch
import torch.nn as nn
from ytch.correctness.gradients import assert_gradients_flow


def test_assert_gradients_flow_passes():
    module = nn.Linear(3, 2)
    x = torch.randn(5, 3, requires_grad=True)
    assert_gradients_flow(module, x)


def test_assert_gradients_flow_frozen_param():
    module = nn.Linear(3, 2)
    module.weight.requires_grad = False
    x = torch.randn(5, 3, requires_grad=True)
    assert_gradients_flow(module, x, check_params={"bias"})


def test_assert_gradients_flow_fails_missing_param_grad():
    class BrokenModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(3, 3))

        def forward(self, x):
            return x @ self.w.detach()

    module = BrokenModule()
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(AssertionError, match="Parameter w gradient is None"):
        assert_gradients_flow(module, x)


def test_assert_gradients_flow_multiple_inputs():
    class DualInput(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(3, 3))

        def forward(self, x, y):
            return x @ self.w + y

    module = DualInput()
    x = torch.randn(2, 3, requires_grad=True)
    y = torch.randn(2, 3, requires_grad=True)
    assert_gradients_flow(module, x, y)


def test_assert_gradients_flow_nonexistent_param():
    module = nn.Linear(3, 2)
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(AssertionError, match="Parameter nonexistent not found"):
        assert_gradients_flow(module, x, check_params={"nonexistent"})


def test_assert_gradients_flow_explicit_frozen_param_fails():
    module = nn.Linear(3, 2)
    module.weight.requires_grad = False
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(
        AssertionError, match="Parameter weight has requires_grad=False"
    ):
        assert_gradients_flow(module, x, check_params={"weight"})
