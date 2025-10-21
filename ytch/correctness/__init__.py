import torch


def assert_shape(x: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    assert x.shape == expected_shape, (
        f"shape mismatch: expected shape {expected_shape}, got {tuple(x.shape)}"
    )
