import math

import torch
from torch import Tensor, nn


class RandomFourierFeaturesND(nn.Module):
    def __init__(self, dims_per_coord: int, sigma=1.0, device=None):
        super().__init__()
        B = sigma * torch.randn(2, dims_per_coord, device=device)
        self.register_buffer("B", B)

    def forward(self, x: Tensor):
        assert isinstance(self.B, Tensor)
        x = 2 * math.pi * x @ self.B
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
