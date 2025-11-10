import torch

from . import grad_multiply

BATCH_SIZE = 2
DIM = 10


def test_grad_multiply_forward():
    """Forward pass returns input unchanged."""
    x = torch.randn(BATCH_SIZE, DIM)
    y = grad_multiply(x, scale=0.5)
    torch.testing.assert_close(y, x)


def test_grad_multiply_scale_zero():
    """scale=0 fully detaches (no gradient flow)."""
    x = torch.randn(BATCH_SIZE, DIM, requires_grad=True)
    y = grad_multiply(x, scale=0.0)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.zeros_like(x))


def test_grad_multiply_scale_one():
    """scale=1 is identity (normal gradient flow)."""
    x = torch.randn(BATCH_SIZE, DIM, requires_grad=True)
    y = grad_multiply(x, scale=1.0)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_grad_multiply_scale_half():
    """scale=0.5 halves gradients."""
    x = torch.randn(BATCH_SIZE, DIM, requires_grad=True)
    y = grad_multiply(x, scale=0.5)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, 0.5 * torch.ones_like(x))
