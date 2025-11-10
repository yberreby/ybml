import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..mh import from_multihead, to_multihead


class CrossAttention(nn.Module):
    """Stateless cross-attention with multihead reshape."""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model={d_model} must be divisible by num_heads={num_heads}"
        )
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

    def forward(
        self,
        q: Float[Tensor, "B N_q D"],
        k: Float[Tensor, "B N_kv D"],
        v: Float[Tensor, "B N_kv D"],
    ) -> Float[Tensor, "B N_q D"]:
        q_heads = to_multihead(q, self.num_heads)
        k_heads = to_multihead(k, self.num_heads)
        v_heads = to_multihead(v, self.num_heads)

        out = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            dropout_p=0.0,
            is_causal=False,
        )
        return from_multihead(out)
