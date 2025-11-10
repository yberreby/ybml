import torch
from ytch.constants import IMAGENET_MEAN, IMAGENET_STD


def test_constants_import():
    """Smoketest: constants can be imported and are tensors."""
    assert isinstance(IMAGENET_MEAN, torch.Tensor)
    assert isinstance(IMAGENET_STD, torch.Tensor)
    assert IMAGENET_MEAN.shape == (3,)
    assert IMAGENET_STD.shape == (3,)
