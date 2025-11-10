import torch
from . import compute_reward_to_go


def test_smoke():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rtg = compute_reward_to_go(rewards)
    assert rtg.shape == (2, 3)
    # First row: [1+2+3, 2+3, 3] = [6, 5, 3]
    assert torch.allclose(rtg[0], torch.tensor([6.0, 5.0, 3.0]))
    # Second row: [4+5+6, 5+6, 6] = [15, 11, 6]
    assert torch.allclose(rtg[1], torch.tensor([15.0, 11.0, 6.0]))
