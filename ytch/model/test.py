import torch.nn as nn
from ytch.model import count_parameters


def test_count_parameters_linear():
    model = nn.Linear(10, 5)
    # 10 * 5 weights + 5 biases = 55
    assert count_parameters(model) == 55


def test_count_parameters_sequential():
    model = nn.Sequential(
        nn.Linear(10, 20),  # 10 * 20 + 20 = 220
        nn.ReLU(),  # 0 params
        nn.Linear(20, 5),  # 20 * 5 + 5 = 105
    )
    # Total: 220 + 0 + 105 = 325
    assert count_parameters(model) == 325


def test_count_parameters_no_bias():
    model = nn.Linear(10, 5, bias=False)
    # 10 * 5 weights, no bias = 50
    assert count_parameters(model) == 50


def test_count_parameters_empty_sequential():
    model = nn.Sequential()
    assert count_parameters(model) == 0
