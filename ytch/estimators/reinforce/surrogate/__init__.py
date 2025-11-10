from torch import Tensor
from jaxtyping import Float


def reinforce_surrogate_loss(
    log_probs: Float[Tensor, "B G"],
    advantages: Float[Tensor, "B G"],
) -> Float[Tensor, ""]:
    """Core REINFORCE surrogate loss: -E[log π(a|s) * A].

    Args:
        log_probs: Log probabilities of actions taken
        advantages: Pre-computed advantages (should already be detached!)

    Returns:
        Scalar loss value
    """
    # Defensive detach: advantages should NOT have gradients in REINFORCE
    # The policy gradient theorem assumes rewards/advantages are independent of θ
    # If advantages accidentally have gradients, we'd compute a biased estimator
    advantages = advantages.detach()

    # Sum over timesteps.
    # Ensure correct sign - this is a loss!
    per_traj_loss = -(log_probs * advantages).sum(dim=1)
    # Take expectation over trajectories
    loss = per_traj_loss.mean()
    return loss
