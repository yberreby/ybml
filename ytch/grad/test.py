import torch
import torch.nn as nn
from torch import Tensor
from ytch.grad import compute_grad_norm


def test_compute_grad_norm():
    model = nn.Linear(3, 2)
    x = torch.randn(5, 3, requires_grad=True)
    y = torch.randn(5, 2)
    criterion = nn.MSELoss()

    loss = criterion(model(x), y)
    loss.backward()

    grad_norm = compute_grad_norm(model)
    assert grad_norm.item() > 0
    assert isinstance(grad_norm, Tensor)


def test_compute_grad_norm_zero_grads():
    model = nn.Linear(3, 2)
    x = torch.randn(5, 3, requires_grad=True)
    y = torch.randn(5, 2)
    criterion = nn.MSELoss()

    loss = criterion(model(x), y)
    loss.backward()
    model.zero_grad()

    grad_norm = compute_grad_norm(model)
    assert grad_norm.item() == 0.0
