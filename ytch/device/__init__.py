import torch


def get_sensible_device(forbid_mps: bool = False) -> torch.device:
    """Get a sensible device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif not forbid_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
