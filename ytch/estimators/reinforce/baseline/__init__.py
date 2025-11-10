from torch import Tensor
from jaxtyping import Float


def compute_leave_one_out_baseline(
    values: Float[Tensor, "Groups N Values"],
) -> Float[Tensor, "Groups N Values"]:
    """
    Compute leave-one-out baseline within each group.

    For each sample in a group, its baseline is the mean of all OTHER
    samples in that group. Groups are isolated.

    Args:
        values: [Groups, N, Values] where:
            - Groups: number of independent groups (use 1 for standard LOO)
            - N: samples per group
            - Values: number of values per sample (e.g., timesteps or 1 for scalar)

    Returns:
        Baselines with same shape as input.
    """
    n = values.shape[1]  # samples per group
    total_sum = values.sum(dim=1, keepdim=True)  # Sum within each group
    # If n is 1, the LOO baseline is not meaningful,
    # so our baseline just ends up being 0 divided by 1.
    return (total_sum - values) / max(n - 1, 1)
