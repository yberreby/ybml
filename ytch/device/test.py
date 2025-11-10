import pytest
import torch

from ytch.device import get_sensible_device, sync_device


def test_get_sensible_device():
    device = get_sensible_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")

    # Verify accepted args
    _ = get_sensible_device(forbid_mps=True)


def test_sync_device_cpu():
    """CPU synchronization should be a no-op."""
    device = torch.device("cpu")
    sync_device(device)  # Should not raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sync_device_cuda():
    """CUDA synchronization should work if CUDA is available."""
    device = torch.device("cuda")
    sync_device(device)  # Should not raise


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_sync_device_mps():
    """MPS synchronization should work if MPS is available."""
    device = torch.device("mps")
    sync_device(device)  # Should not raise


def test_sync_device_unknown():
    """Unknown device types should raise ValueError."""

    # Create a mock device-like object with an unknown type
    class FakeDevice:
        def __init__(self, device_type: str):
            self.type = device_type

    fake_device = FakeDevice("xpu")
    with pytest.raises(ValueError, match="Unsupported device type"):
        sync_device(fake_device)  # pyright: ignore[reportArgumentType]
