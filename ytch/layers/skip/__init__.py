import torch.nn as nn
from torch import Tensor


class Skip(nn.Module):
    """Residual connection: outputs x + block(x)."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)

    def __repr__(self) -> str:
        return f"Skip({self.block})"
