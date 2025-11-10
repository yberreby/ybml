import torch
from torch import Tensor
from jaxtyping import Float


def get_loss_reinforce(
    per_glimpse_reward: Float[Tensor, "B G"],
    log_probs: Float[Tensor, "B G"],
) -> Float[Tensor, ""]:
    # CRITICAL: For correct REINFORCE, rewards MUST be detached.
    # REINFORCE assumes rewards are independent of the policy parameters θ.
    # If rewards depend on θ (e.g., through learned predictions), failing to
    # detach creates a biased estimator with an extra gradient term: log_prob · ∇_θ R
    per_glimpse_reward = per_glimpse_reward.detach()

    # Compute the future reward for each glimpse
    rewards_reversed = torch.flip(per_glimpse_reward, dims=(1,))
    future_reward = torch.cumsum(rewards_reversed, dim=1)
    future_reward = torch.flip(future_reward, dims=(1,))

    loss_per_glimpse = -log_probs * future_reward
    loss_reinforce = torch.sum(loss_per_glimpse, dim=1).mean()

    return loss_reinforce
