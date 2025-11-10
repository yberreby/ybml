import torch
from . import FiLM


def test_film_vector():
    """Test FiLM on vector inputs."""
    film = FiLM(cond_dim=16, channels=8)
    x = torch.randn(2, 8)
    cond = torch.randn(2, 16)
    y = film(x, cond)
    assert y.shape == (2, 8)


def test_film_feature_map():
    """Test FiLM on 2D feature maps."""
    film = FiLM(cond_dim=16, channels=8)
    x = torch.randn(2, 8, 4, 4)
    cond = torch.randn(2, 16)
    y = film(x, cond)
    assert y.shape == (2, 8, 4, 4)


def test_film_identity_init():
    """Test that FiLM starts as identity."""
    film = FiLM(cond_dim=16, channels=8)
    x = torch.randn(2, 8)
    cond = torch.randn(2, 16)
    y = film(x, cond)
    torch.testing.assert_close(y, x, rtol=1e-5, atol=1e-5)
