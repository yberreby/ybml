import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Canonical Feature-wise Linear Modulation (FiLM), linear generator only.
    y = (1 + Δγ(cond)) ⊙ x + Δβ(cond)
    - x:    [B, C, *spatial]   (vector: [B, C]; feature map: [B, C, H, W]; etc.)
    - cond: [B, E]             (e.g., class embedding or RNN state)
    Identity init (Δγ=0, Δβ=0) -> safe to drop into pretrained nets.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.channels = channels
        self.gen = nn.Linear(cond_dim, 2 * channels, bias=True)
        # Identity init: first step does nothing (gamma=1, beta=0)
        nn.init.zeros_(self.gen.weight)
        nn.init.zeros_(self.gen.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert x.dim() >= 2, f"FiLM expects [B,C,...], got {tuple(x.shape)}"
        B, C = x.shape[0], x.shape[1]
        assert C == self.channels, f"channels mismatch: {C=} vs {self.channels=}"
        assert cond.shape[0] == B, f"batch mismatch: {cond.shape[0]=} vs {B=}"
        assert cond.dim() == 2, f"cond must be [B,E], got {tuple(cond.shape)}"

        dgamma, dbeta = self.gen(cond).chunk(2, dim=1)  # [B,C], [B,C]
        bshape = (B, C) + (1,) * (x.ndim - 2)
        gamma = (1.0 + dgamma).view(bshape)
        beta = dbeta.view(bshape)
        return gamma * x + beta
