import torch


def get_sensible_device(forbid_mps: bool = False) -> torch.device:
    """Get a sensible device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif not forbid_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Synchronize device operations for accurate timing/benchmarking.

    CPU operations are synchronous by default and require no sync.
    Raises ValueError for unsupported device types.
    """
    if device.type == "cpu":
        return  # CPU operations are synchronous by default
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    else:
        raise ValueError(
            f"Unsupported device type: {device.type!r}. Supported: 'cpu', 'cuda', 'mps'"
        )
