from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


def _init_output_layer(model: "SimpleMLP", target_logit_std: float = 0.2) -> None:
    """Initialize output layer based on measured activation statistics."""
    hidden_dim = model.fc_out.in_features
    n_samples = 10 * hidden_dim

    with torch.no_grad():
        x_sample = torch.randn(n_samples, 2)
        h = model._backbone(x_sample)

        # Compute total variance scale: √(Σ var(h_i))
        per_dim_var = h.var(dim=0, unbiased=False)
        total_var_scale = per_dim_var.sum().sqrt().item()

        # Set init std so that logit_std ≈ target_logit_std
        output_init_std = target_logit_std / total_var_scale

    nn.init.normal_(model.fc_out.weight, std=output_init_std)
    nn.init.zeros_(model.fc_out.bias)


class SimpleMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        smart_output_init: bool = True,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
    ):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 2)
        if smart_output_init:
            _init_output_layer(self)

    def _backbone(self, x: Tensor) -> Tensor:
        """Backbone: everything before output layer."""
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.fc_out(self._backbone(x))


class TwoMoonsClassifier(nn.Module):
    """SimpleMLP with CrossEntropyLoss for training."""

    def __init__(self, hidden_dim: int, smart_output_init: bool = True):
        super().__init__()
        self.mlp = SimpleMLP(hidden_dim, smart_output_init=smart_output_init)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        logits = self.mlp(x)
        loss = self.criterion(logits, y)
        return {"loss": loss, "logits": logits}
