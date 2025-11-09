from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from ymc.lr import get_warmup_steps_for_adam_beta2


def get_linear_warmup_scheduler(
    optimizer: Adam | AdamW, warmup_steps: int | None = None
) -> LambdaLR:
    """Linear warmup scheduler tuned to each param group's beta2."""
    lr_lambdas = []
    for pg in optimizer.param_groups:
        _beta1, beta2 = pg.get("betas", optimizer.defaults["betas"])
        if warmup_steps is None:
            ws_for_group = get_warmup_steps_for_adam_beta2(beta2)
        else:
            ws_for_group = warmup_steps
        f = lambda step, ws=ws_for_group: min(1.0, (step + 1) / ws)
        lr_lambdas.append(f)
    return LambdaLR(optimizer, lr_lambda=lr_lambdas)
