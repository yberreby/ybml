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


class TwoMoonsClassifier(nn.Module):
    """SimpleMLP with CrossEntropyLoss for training."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = SimpleMLP(hidden_dim)
        self.criterion = nn.CrossEntropyLoss()

    @override
    def forward(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        logits = self.mlp(x)
        loss = self.criterion(logits, y)
        return {"loss": loss, "logits": logits}
