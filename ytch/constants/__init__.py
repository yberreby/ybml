import torch

from ymc.constants import IMAGENET_MEAN as IMAGENET_MEAN_PY
from ymc.constants import IMAGENET_STD as IMAGENET_STD_PY

IMAGENET_MEAN = torch.tensor(IMAGENET_MEAN_PY)
IMAGENET_STD = torch.tensor(IMAGENET_STD_PY)
