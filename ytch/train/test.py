import torch
import torch.nn as nn

from ytch.train import train


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(2, 1)

    def forward(self, x, y):
        return {"loss": ((self.w(x) - y) ** 2).mean()}


def test_train_smoke():
    model = TinyModel()
    data = lambda: (torch.randn(8, 2), torch.randn(8, 1))
    result = train(model, data, n_steps=10, lr=1e-3)

    assert "loss" in result
    assert "lr" in result
    assert "gnpre" in result
    assert "samples/s" in result
