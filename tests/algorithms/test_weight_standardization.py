# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.utils.parametrize as parametrize

from composer.algorithms import WeightStandardization
from composer.core import Event, State
from composer.functional import apply_weight_standardization
from composer.loggers import Logger
from tests.common import SimpleConvModel


def _count_parametrize(module: torch.nn.Module):
    count = 0
    for m in module.modules():
        if parametrize.is_parametrized(m, 'weight'):
            count += 1
    return count


def test_ws_calculation():
    """Check if convolution weights are properly standardized."""
    model = SimpleConvModel()
    apply_weight_standardization(module=model)
    var1, mean1 = torch.var_mean(model.conv1.weight, dim=[1, 2, 3], unbiased=False)
    var2, mean2 = torch.var_mean(model.conv2.weight, dim=[1, 2, 3], unbiased=False)
    torch.testing.assert_close(var1, torch.ones_like(var1))
    torch.testing.assert_close(var2, torch.ones_like(var2))
    torch.testing.assert_close(mean1, torch.zeros_like(mean1))
    torch.testing.assert_close(mean2, torch.zeros_like(mean2))


@pytest.mark.parametrize('n_last_layers_ignore', [0, 1, 3])
def test_ws_replacement(n_last_layers_ignore: int):
    """Check if the proper number of layers have been parametrized."""
    model = SimpleConvModel()
    apply_weight_standardization(module=model, n_last_layers_ignore=n_last_layers_ignore)
    ws_count = _count_parametrize(model)
    expected_count = max(2 - n_last_layers_ignore, 0)  # Expected number of weight standardization layers
    assert ws_count == expected_count


@pytest.mark.parametrize('n_last_layers_ignore', [0, 1, 3])
def test_ws_algorithm(n_last_layers_ignore: int, minimal_state: State, empty_logger: Logger):
    """Check if the algorithm is applied at the proper event."""
    minimal_state.model = SimpleConvModel()
    ws_algorithm = WeightStandardization(n_last_layers_ignore=n_last_layers_ignore)
    ws_algorithm.apply(Event.INIT, minimal_state, empty_logger)

    ws_count = _count_parametrize(minimal_state.model)
    expected_count = max(2 - n_last_layers_ignore, 0)
    assert ws_count == expected_count
