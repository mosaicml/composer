# Copyright 2021 MosaicML. All Rights Reserved.

import pytest
import torch

from composer.algorithms import Factorize, FactorizeHparams
from composer.algorithms.factorize import FactorizedConv2d
from composer.algorithms.factorize.factorize import FACTORIZE_LOG_NUM_REPLACEMENTS_KEY
from composer.core import Event, Logger, State, surgery
from composer.core.types import Tensors
from composer.trainer.trainer_hparams import TrainerHparams
from tests.utils.trainer_fit import train_model


def _apply_algo(state_with_model: State, simple_conv_model_input: Tensors, logger: Logger):
    batch = (simple_conv_model_input, None)
    out = state_with_model.model.forward(batch)
    original_size = out.size()
    original_conv_count = surgery.count_module_instances(state_with_model.model, torch.nn.Conv2d)

    algo = Factorize(latent_channels=2, min_channels=3)
    algo.apply(
        event=Event.INIT,
        state=state_with_model,
        logger=logger,
    )

    return original_conv_count, original_size


def test_layer_replacement(state_with_model: State, simple_conv_model_input: Tensors, noop_dummy_logger: Logger):
    original_conv_count, _ = _apply_algo(
        state_with_model,
        simple_conv_model_input,
        noop_dummy_logger,
    )

    # verify layer replacement
    replace_count = surgery.count_module_instances(state_with_model.model, FactorizedConv2d)
    assert original_conv_count == replace_count


def test_forward_shape(state_with_model: State, simple_conv_model_input: Tensors, noop_dummy_logger: Logger):
    _, original_size = _apply_algo(
        state_with_model,
        simple_conv_model_input,
        noop_dummy_logger,
    )

    # verify forward prop still works
    batch = (simple_conv_model_input, torch.Tensor())
    out = state_with_model.model.forward(batch)
    assert original_size == out.size()


def test_algorithm_logging(state_with_model: State, logger_mock: Logger):
    algo = Factorize(latent_channels=2, min_channels=4)
    conv_count = surgery.count_module_instances(state_with_model.model, torch.nn.Conv2d)
    algo.apply(Event.INIT, state_with_model, logger=logger_mock)

    logger_mock.metric_fit.assert_called_once_with({
        FACTORIZE_LOG_NUM_REPLACEMENTS_KEY: conv_count,
    })


# not marked run_long because takes a fraction of a second
def test_factorize_trains(mosaic_trainer_hparams: TrainerHparams):
    mosaic_trainer_hparams.algorithms = [FactorizeHparams(latent_channels=4, min_channels=8)]
    train_model(mosaic_trainer_hparams, run_loss_check=True)
