import torch

from ytch.correctness import assert_gradients_flow

from . import CrossAttention


def test_cross_attention_shape():
    B, N_q, N_kv, D, H = 2, 10, 20, 64, 8
    ca = CrossAttention(num_heads=H, d_model=D)
    q = torch.randn(B, N_q, D, requires_grad=True)
    k = torch.randn(B, N_kv, D, requires_grad=True)
    v = torch.randn(B, N_kv, D, requires_grad=True)
    out = ca(q, k, v)
    assert out.shape == (B, N_q, D)


def test_cross_attention_gradients():
    """Gradients flow through q, k, v."""
    B, N_q, N_kv, D, H = 2, 10, 20, 64, 8
    ca = CrossAttention(num_heads=H, d_model=D)
    q = torch.randn(B, N_q, D, requires_grad=True)
    k = torch.randn(B, N_kv, D, requires_grad=True)
    v = torch.randn(B, N_kv, D, requires_grad=True)
    assert_gradients_flow(ca, q, k, v)
