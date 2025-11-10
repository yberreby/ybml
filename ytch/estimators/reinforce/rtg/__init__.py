import torch
from torch import Tensor
from jaxtyping import Float


def compute_reward_to_go(
    per_glimpse_reward: Float[Tensor, "... G"],
) -> Float[Tensor, "... G"]:
    """Convert per-step rewards to reward-to-go (G_t).

    G_t = sum_{k=t}^{T-1} r_k

    Works on the last dimension, so supports both [B, G] and [Groups, N, G].
    """
    rewards_reversed = torch.flip(per_glimpse_reward, dims=(-1,))
    future_reward = torch.cumsum(rewards_reversed, dim=-1)
    future_reward = torch.flip(future_reward, dims=(-1,))
    return future_reward
