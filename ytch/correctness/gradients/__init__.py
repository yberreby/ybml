import torch
import torch.nn as nn


def assert_gradients_flow(
    module: nn.Module, *inputs: torch.Tensor, check_params: set[str] | None = None
) -> None:
    """Assert gradients flow to inputs and parameters.

    All inputs must have requires_grad=True.

    Args:
        module: Module to test
        *inputs: Input tensors to check (all must have requires_grad=True)
        check_params: Param names to check. If None, checks all params with requires_grad=True.
    """
    for i, inp in enumerate(inputs):
        assert inp.requires_grad, f"Input {i} must have requires_grad=True"

    output = module(*inputs)
    assert isinstance(output, torch.Tensor), "Module must return a single Tensor"
    output.sum().backward()

    for i, inp in enumerate(inputs):
        assert inp.grad is not None, f"Input {i} gradient is None"

    param_dict = dict(module.named_parameters())
    params_to_check = (
        check_params if check_params is not None else set(param_dict.keys())
    )

    for name in params_to_check:
        assert name in param_dict, f"Parameter {name} not found in module"
        param = param_dict[name]

        if not param.requires_grad:
            if check_params is not None:
                raise AssertionError(f"Parameter {name} has requires_grad=False")
            continue

        assert param.grad is not None, f"Parameter {name} gradient is None"
