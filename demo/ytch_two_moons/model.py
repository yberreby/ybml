from typing import override

import torch.nn as nn
from torch import Tensor


class SimpleMLP(nn.Module):
    net: nn.Sequential

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
