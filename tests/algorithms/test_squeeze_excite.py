# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from composer.algorithms import SqueezeExcite, SqueezeExciteConv2d
from composer.core import Event, State
from composer.functional import apply_squeeze_excite as apply_se
from composer.loggers import Logger
from composer.utils import module_surgery
from tests.common import SimpleConvModel


@pytest.fixture
def state(minimal_state: State):
    """SE tests require a conv model."""
    minimal_state.model = SimpleConvModel(num_channels=32)
    return minimal_state


def test_se_functional():
    model = SimpleConvModel()
    num_conv_layers = module_surgery.count_module_instances(model, torch.nn.Conv2d)
    apply_se(model, latent_channels=64, min_channels=3)
    num_se_layers = module_surgery.count_module_instances(model, SqueezeExciteConv2d)

    assert num_conv_layers == num_se_layers


def test_se_algorithm(state: State, empty_logger: Logger):
    num_conv_layers = module_surgery.count_module_instances(state.model, torch.nn.Conv2d)

    algorithm = SqueezeExcite(latent_channels=64, min_channels=3)
    algorithm.apply(
        event=Event.INIT,
        state=state,
        logger=empty_logger,
    )

    num_se_layers = module_surgery.count_module_instances(state.model, SqueezeExciteConv2d)
    assert num_conv_layers == num_se_layers


def test_se_logging(state: State, empty_logger: Logger):
    logger_mock = Mock()

    se = SqueezeExcite(latent_channels=64, min_channels=3)
    se.apply(Event.INIT, state, logger=logger_mock)
    conv_count = module_surgery.count_module_instances(state.model, torch.nn.Conv2d)

    logger_mock.log_hyperparameters.assert_called_once_with({
        'squeeze_excite/num_squeeze_excite_layers': conv_count,
    })


def test_se_forward_shape(state: State):
    batch = (torch.Tensor(8, 32, 64, 64), None)  # NCHW
    output = state.model.forward(batch)

    apply_se(state.model, latent_channels=32, min_channels=3)

    new_output = state.model.forward(batch)
    assert output.size() == new_output.size()
