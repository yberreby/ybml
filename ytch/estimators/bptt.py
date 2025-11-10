import torch
from torch import Tensor
from jaxtyping import Float


def get_loss_bptt(
    per_glimpse_reward: Float[Tensor, "Groups N G"], *args
) -> Float[Tensor, ""]:
    # Note: this is commented in excruciating detail, by hand, because details matter.
    # Do not "simplify" the comments or rewrite them without careful pondering!
    # Critical functions such as this one, even if the code is super simple (3 operations!)
    # should be made so crystal clear that it feels excessive.
    # It should be almost impossible to make a sign flip error,
    # or to gloss over a mathematically-invalid operation.

    # BPTT doesn't use groups - flatten to [B, G] where B = Groups * N
    groups, n, g = per_glimpse_reward.shape
    per_glimpse_reward_flat: Float[Tensor, "B G"] = per_glimpse_reward.reshape(-1, g)

    # Each episode has `G` time steps (recall that G is the number of glimpses
    # and does not include the inital imposed/random glimpse)
    # Matching the standard RL problem setting, we compute the *cumulative reward*
    # for each of the `B` episodes independently.
    per_ep_cumulative_reward: Float[Tensor, "B"] = torch.sum(
        per_glimpse_reward_flat, dim=1
    )

    # We're given a reward, which is supposed to be maximized.
    # But we output a *loss* (or "penalty"), which will be minimized.
    # Thus, we flip sign for minimization vs maximization.
    # This is always principled independently of context,
    # since `argmin -f = argmax f`
    #
    # Here we use the `task_penalty` terminology to make it easy to distinguish
    # what comes from the task's reward structure,
    # and what might (later) come from e.g. entropy regularization.
    #
    # Entropy regularization is *task-agnostic* and would thus not be in this term.
    # For now, no entropy regularization, but it will come.
    per_ep_task_penalty: Float[Tensor, "B"] = -per_ep_cumulative_reward

    # Now we simply take the average across the batch, and get a scalar tensor back.
    #
    # The point here is to compute the *gradient of the expected cumulative reward*.
    # The "gradient" part comes from letting autograd do the job.
    # What autograd is really going to compute is "the gradient of the average cumulative reward across the batch".
    # Note that this only works because our gradients are well-behaved.
    # In cases where they aren't, it is not principled to let an *expectation* and a *gradient* commute,
    # which is what's implicitly what's happening here!
    batch_avg_penalty: Float[Tensor, ""] = per_ep_task_penalty.mean()

    # The loss is just the average penalty across batch for now; add regularization terms later.
    return batch_avg_penalty
