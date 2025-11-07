import torch
import torch.nn as nn


# Relevant: "Initialization of ReLUs for Dynamical Isometry" (Burkholz and Dubatovka, 2019)
# https://proceedings.neurips.cc/paper/2019/file/d9731321ef4e063ebbee79298fa36f56-Paper.pdf
#
# Note:
# - Meant for ReLU
# - The "parameter sharing" is ONLY at initialization, all parameters are free after.
def ortho_block_init_(layer: nn.Linear):
    """
    Weight-shared linear layer initialization with random orthogonal W0 template.

    With this,
    you can compose (Linear o ReLU) blocks to high depths,
    even without skip connections,
    and still be able to train your model without vanishing/exploding gradients.

    Empirically, this seems to work OK with SiLU as well.
    """

    assert layer.out_features % 2 == 0 and layer.in_features % 2 == 0, (
        "Layer input/output dimensions must be even"
    )
    h_out, h_in = layer.out_features // 2, layer.in_features // 2
    with torch.no_grad():
        # Sample orthogonal template.
        W0 = torch.nn.init.orthogonal_(torch.empty(h_out, h_in))

        # Apply 2x2 block structure.
        layer.weight.data[:h_out, :h_in] = W0
        layer.weight.data[:h_out, h_in:] = -W0
        layer.weight.data[h_out:, :h_in] = -W0
        layer.weight.data[h_out:, h_in:] = W0

        # Zero out bias.
        _ = layer.bias.data.zero_()
