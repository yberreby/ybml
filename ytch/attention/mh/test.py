import torch

from . import from_multihead, to_multihead


def test_to_multihead():
    B, N, D, H = 2, 10, 64, 8
    x = torch.randn(B, N, D)
    out = to_multihead(x, num_heads=H)
    assert out.shape == (B, H, N, D // H)


def test_multihead_roundtrip():
    B, N, D, H = 2, 10, 64, 8
    x = torch.randn(B, N, D)
    out = from_multihead(to_multihead(x, num_heads=H))
    assert out.shape == x.shape
    assert torch.allclose(out, x)
