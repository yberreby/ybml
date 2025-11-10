from torch import Tensor
from jaxtyping import Float

from .surrogate import reinforce_surrogate_loss
from .rtg import compute_reward_to_go
from .baseline import compute_leave_one_out_baseline


def get_loss_reinforce(
    per_glimpse_reward: Float[Tensor, "Groups N G"],
    log_probs: Float[Tensor, "Groups N G"],
) -> Float[Tensor, ""]:
    """
    Vanilla REINFORCE: no baseline, advantages = G_t.
    This is _really_ bad and should never be used for serious stuff,
    but it's an important comparison point.

    Use Groups=1 for standard single-group batch.
    """
    groups, n, g = per_glimpse_reward.shape
    future_reward = compute_reward_to_go(per_glimpse_reward)

    # Flatten for loss computation
    log_probs_flat = log_probs.reshape(-1, g)
    advantages_flat = future_reward.reshape(-1, g)
    return reinforce_surrogate_loss(log_probs_flat, advantages=advantages_flat)


def get_loss_rloo_stepwise(
    per_glimpse_reward: Float[Tensor, "Groups N G"],
    log_probs: Float[Tensor, "Groups N G"],
) -> Float[Tensor, ""]:
    """
    Stepwise RLOO: baseline computed per timestep within groups.

    Groups are isolated - samples in different groups don't affect
    each other's baselines. Use Groups=1 for standard RLOO across batch.
    """
    groups, n, g = per_glimpse_reward.shape
    future_reward = compute_reward_to_go(per_glimpse_reward)
    baseline = compute_leave_one_out_baseline(future_reward)
    advantages = future_reward - baseline

    # Flatten for loss computation
    log_probs_flat = log_probs.reshape(-1, g)
    advantages_flat = advantages.reshape(-1, g)
    return reinforce_surrogate_loss(log_probs_flat, advantages_flat)


def get_loss_rloo_scalar(
    per_glimpse_reward: Float[Tensor, "Groups N G"],
    log_probs: Float[Tensor, "Groups N G"],
) -> Float[Tensor, ""]:
    """
    Scalar RLOO: single baseline per trajectory within groups.

    Each trajectory gets one scalar baseline from other trajectories
    in its group. Generally worse than stepwise version.
    """
    groups, n, g = per_glimpse_reward.shape
    future_reward = compute_reward_to_go(per_glimpse_reward)

    # Total trajectory reward is G_0 (reward-to-go at timestep 0)
    total_rewards = future_reward[:, :, 0:1]  # [Groups, N, 1]

    # Compute LOO baseline from total rewards within groups
    scalar_baseline = compute_leave_one_out_baseline(total_rewards)

    # Broadcast single baseline to all timesteps
    baseline = scalar_baseline.expand_as(future_reward)
    advantages = future_reward - baseline

    # Flatten for loss computation
    log_probs_flat = log_probs.reshape(-1, g)
    advantages_flat = advantages.reshape(-1, g)
    return reinforce_surrogate_loss(log_probs_flat, advantages_flat)
