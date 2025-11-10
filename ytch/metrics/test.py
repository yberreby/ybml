import torch
import torch.nn as nn
from torch import Tensor
from ytch.metrics import compute_grad_norm, print_grad_norms


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


def test_print_grad_norms():
    model = nn.Linear(3, 2)
    x = torch.randn(5, 3, requires_grad=True)
    y = torch.randn(5, 2)
    criterion = nn.MSELoss()

    loss = criterion(model(x), y)
    loss.backward()

    print_grad_norms(model)
    print_grad_norms(model, prefix="test")


def test_print_grad_norms_with_none_grad(capsys):
    """Smoketest: print_grad_norms handles None gradients."""
    model = nn.Linear(3, 2)
    model.weight.requires_grad = False
    x = torch.randn(5, 3, requires_grad=True)
    y = torch.randn(5, 2)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()

    print_grad_norms(model)
    captured = capsys.readouterr()
    assert "None" in captured.out
