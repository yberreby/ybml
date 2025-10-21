import torch
from ytch.device import get_sensible_device


def test_get_sensible_device():
    device = get_sensible_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")
