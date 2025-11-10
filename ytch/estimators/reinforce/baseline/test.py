import torch
from . import compute_leave_one_out_baseline


def test_smoke():
    # Test with Groups=1, N=3, Values=2
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    baseline = compute_leave_one_out_baseline(values)
    assert baseline.shape == (1, 3, 2)

    # For each position, baseline is mean of other two
    # Position 0: mean of [[3,4], [5,6]] = [4, 5]
    assert torch.allclose(baseline[0, 0], torch.tensor([4.0, 5.0]))
    # Position 1: mean of [[1,2], [5,6]] = [3, 4]
    assert torch.allclose(baseline[0, 1], torch.tensor([3.0, 4.0]))
    # Position 2: mean of [[1,2], [3,4]] = [2, 3]
    assert torch.allclose(baseline[0, 2], torch.tensor([2.0, 3.0]))


def test_grouped():
    # Test with Groups=2, N=2, Values=1
    values = torch.tensor(
        [
            [[1.0], [3.0]],  # Group 0
            [[5.0], [7.0]],  # Group 1
        ]
    )
    baseline = compute_leave_one_out_baseline(values)
    assert baseline.shape == (2, 2, 1)

    # Group 0: each gets the other as baseline
    assert torch.allclose(baseline[0, 0], torch.tensor([3.0]))
    assert torch.allclose(baseline[0, 1], torch.tensor([1.0]))

    # Group 1: each gets the other as baseline
    assert torch.allclose(baseline[1, 0], torch.tensor([7.0]))
    assert torch.allclose(baseline[1, 1], torch.tensor([5.0]))
