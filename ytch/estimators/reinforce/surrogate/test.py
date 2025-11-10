import torch
from . import reinforce_surrogate_loss


def test_smoke():
    log_probs = torch.tensor([[-1.0, -2.0], [-0.5, -1.5]], requires_grad=True)
    advantages = torch.tensor([[1.0, 0.5], [2.0, 1.0]])
    loss = reinforce_surrogate_loss(log_probs, advantages)
    assert loss.shape == ()
    # Mean of [(1.0 + 1.0), (1.0 + 1.5)] = mean of [2.0, 2.5] = 2.25
    assert torch.allclose(loss, torch.tensor(2.25))
