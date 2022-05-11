# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import itertools
from unittest.mock import Mock

import pytest
import torch

from composer.algorithms import Factorize
from composer.algorithms.factorize import FactorizedConv2d, FactorizedLinear
from composer.algorithms.factorize.factorize import LOG_NUM_CONV2D_REPLACEMENTS_KEY, LOG_NUM_LINEAR_REPLACEMENTS_KEY
from composer.core import Event, State
from composer.loggers import Logger
from composer.utils import module_surgery
from tests.common import SimpleConvModel


@pytest.fixture
def state(minimal_state: State):
    minimal_state.model = SimpleConvModel(num_classes=100)

    return minimal_state


@pytest.fixture(params=itertools.product((False, True), (False, True)))
def algo_instance(request):
    replace_convs, replace_linears = request.param
    return Factorize(factorize_convs=replace_convs,
                     factorize_linears=replace_linears,
                     min_channels=1,
                     latent_channels=2,
                     min_features=1,
                     latent_features=2)


@pytest.fixture()
def simple_conv_model_input():
    return torch.rand((64, 32, 64, 64))


def test_factorize_surgery(state: State, empty_logger: Logger, algo_instance: Factorize):

    num_conv_layers = module_surgery.count_module_instances(state.model, torch.nn.Conv2d)
    num_linear_layers = module_surgery.count_module_instances(state.model, torch.nn.Linear)

    algo_instance.apply(event=Event.INIT, state=state, logger=empty_logger)

    if algo_instance.factorize_convs:
        num_factorized_layers = module_surgery.count_module_instances(state.model, FactorizedConv2d)
        assert num_conv_layers == num_factorized_layers
        assert num_factorized_layers > 0

    if algo_instance.factorize_linears:
        num_factorized_layers = module_surgery.count_module_instances(state.model, FactorizedLinear)
        assert num_linear_layers == num_factorized_layers
        assert num_factorized_layers > 0


def test_forward_shape(state: State, empty_logger: Logger, algo_instance: Factorize):

    batch = (torch.Tensor(64, 3, 32, 32), torch.Tensor())
    output = state.model.forward(batch)

    algo_instance.apply(event=Event.INIT, state=state, logger=empty_logger)
    new_output = state.model.forward(batch)

    assert output.size() == new_output.size()


def test_algorithm_logging(state: State, algo_instance: Factorize):
    logger_mock = Mock()

    conv_count = module_surgery.count_module_instances(state.model, torch.nn.Conv2d)
    linear_count = module_surgery.count_module_instances(state.model, torch.nn.Linear)
    algo_instance.apply(Event.INIT, state, logger=logger_mock)

    factorize_convs = algo_instance.factorize_convs
    factorize_linears = algo_instance.factorize_linears
    mock_obj = logger_mock.data_fit

    if factorize_convs:
        mock_obj.assert_any_call({LOG_NUM_CONV2D_REPLACEMENTS_KEY: conv_count})
    if factorize_linears:
        mock_obj.assert_any_call({LOG_NUM_LINEAR_REPLACEMENTS_KEY: linear_count})

    target_count = 0
    target_count += 1 if factorize_convs else 0
    target_count += 1 if factorize_linears else 0
    assert mock_obj.call_count == target_count
