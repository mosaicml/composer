# Copyright 2021 MosaicML. All Rights Reserved.

import itertools

import pytest
import torch

from composer.algorithms import Factorize, FactorizeHparams
from composer.algorithms.factorize import FactorizedConv2d, FactorizedLinear
from composer.algorithms.factorize.factorize import LOG_NUM_CONV2D_REPLACEMENTS_KEY, LOG_NUM_LINEAR_REPLACEMENTS_KEY
from composer.core import Event, Logger, State, surgery
from composer.core.algorithm import Algorithm
from composer.core.types import Tensors
from composer.trainer.trainer_hparams import TrainerHparams
from tests.utils.trainer_fit import train_model


@pytest.fixture(params=itertools.product((False, True), (False, True)))
def algo_instance(request):
    replace_convs, replace_linears = request.param
    return Factorize(factorize_convs=replace_convs,
                     factorize_linears=replace_linears,
                     min_channels=3,
                     latent_channels=2,
                     min_features=8,
                     latent_features=4)


def _apply_algo(state_with_model: State, simple_conv_model_input: Tensors, algo_instance: Algorithm, logger: Logger):
    batch = (simple_conv_model_input, None)
    original_conv_count = surgery.count_module_instances(state_with_model.model, torch.nn.Conv2d)
    original_linear_count = surgery.count_module_instances(state_with_model.model, torch.nn.Linear)
    out = state_with_model.model.forward(batch)
    original_shape = out.shape

    algo_instance.apply(
        event=Event.INIT,
        state=state_with_model,
        logger=logger,
    )

    return original_conv_count, original_linear_count, original_shape


def test_layer_replacement(state_with_model: State, simple_conv_model_input: Tensors, noop_dummy_logger: Logger,
                           algo_instance: Factorize):
    original_conv_count, original_linear_count, _ = _apply_algo(
        state_with_model=state_with_model,
        simple_conv_model_input=simple_conv_model_input,
        algo_instance=algo_instance,
        logger=noop_dummy_logger,
    )

    # verify that layer replacements have happened
    if algo_instance.factorize_convs:
        assert original_conv_count == surgery.count_module_instances(state_with_model.model, FactorizedConv2d)
    if algo_instance.factorize_linears:
        assert original_linear_count == surgery.count_module_instances(state_with_model.model, FactorizedLinear)


def test_forward_shape(state_with_model: State, simple_conv_model_input: Tensors, noop_dummy_logger: Logger,
                       algo_instance: Factorize):
    _, _, original_shape = _apply_algo(
        state_with_model=state_with_model,
        simple_conv_model_input=simple_conv_model_input,
        algo_instance=algo_instance,
        logger=noop_dummy_logger,
    )
    batch = (simple_conv_model_input, torch.Tensor())
    out = state_with_model.model.forward(batch)
    assert original_shape == out.size()


def test_algorithm_logging(state_with_model: State, logger_mock: Logger, algo_instance: Factorize):
    conv_count = surgery.count_module_instances(state_with_model.model, torch.nn.Conv2d)
    linear_count = surgery.count_module_instances(state_with_model.model, torch.nn.Linear)
    algo_instance.apply(Event.INIT, state_with_model, logger=logger_mock)

    factorize_convs = algo_instance.factorize_convs
    factorize_linears = algo_instance.factorize_linears
    mock_obj = logger_mock.metric_fit

    if factorize_convs:
        mock_obj.assert_any_call({LOG_NUM_CONV2D_REPLACEMENTS_KEY: conv_count})
    if factorize_linears:
        mock_obj.assert_any_call({LOG_NUM_LINEAR_REPLACEMENTS_KEY: linear_count})

    target_count = 0
    target_count += 1 if factorize_convs else 0
    target_count += 1 if factorize_linears else 0
    assert mock_obj.call_count == target_count


# not marked run_long because takes a fraction of a second
def test_factorize_trains(mosaic_trainer_hparams: TrainerHparams):
    mosaic_trainer_hparams.algorithms = [
        FactorizeHparams(factorize_convs=True,
                         factorize_linears=True,
                         min_channels=8,
                         latent_channels=4,
                         min_features=8,
                         latent_features=4)
    ]
    train_model(mosaic_trainer_hparams, run_loss_check=True)
