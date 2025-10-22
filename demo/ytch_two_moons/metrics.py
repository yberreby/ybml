from torch import Tensor


def compute_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Compute classification accuracy from logits and targets.

    Args:
        logits: Model output logits (batch_size, num_classes)
        targets: Ground truth labels (batch_size,)

    Returns:
        Accuracy as a float in [0, 1]
    """
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()
