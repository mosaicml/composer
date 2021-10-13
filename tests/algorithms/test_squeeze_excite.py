# Copyright 2021 MosaicML. All Rights Reserved.

import pytest
import torch

from composer.algorithms import SqueezeExcite, SqueezeExciteConv2d
from composer.algorithms.squeeze_excite.squeeze_excite import SqueezeExciteHparams
from composer.core import Event, Logger, State, surgery
from composer.core.types import Tensors
from composer.trainer.trainer_hparams import TrainerHparams
from tests.utils.trainer_fit import train_model


def _do_squeeze_excite(state_with_model: State, simple_conv_model_input: Tensors, logger: Logger):
    batch = (simple_conv_model_input, None)
    out = state_with_model.model.forward(batch)
    original_size = out.size()
    original_conv_count = surgery.count_module_instances(state_with_model.model, torch.nn.Conv2d)

    se = SqueezeExcite(latent_channels=64, min_channels=3)
    se.apply(
        event=Event.INIT,
        state=state_with_model,
        logger=logger,
    )

    return original_conv_count, original_size


def test_squeeze_excite_layer_replacement(state_with_model: State, simple_conv_model_input: Tensors,
                                          noop_dummy_logger: Logger):
    original_conv_count, _ = _do_squeeze_excite(
        state_with_model,
        simple_conv_model_input,
        noop_dummy_logger,
    )

    # verify layer replacement
    se_count = surgery.count_module_instances(state_with_model.model, SqueezeExciteConv2d)
    assert original_conv_count == se_count


def test_squeeze_excite_forward_shape(state_with_model: State, simple_conv_model_input: Tensors,
                                      noop_dummy_logger: Logger):
    _, original_size = _do_squeeze_excite(
        state_with_model,
        simple_conv_model_input,
        noop_dummy_logger,
    )

    # verify forward prop still works
    batch = (simple_conv_model_input, torch.Tensor())
    out = state_with_model.model.forward(batch)
    post_se_size = out.size()
    assert original_size == post_se_size


def test_squeeze_excite_algorithm_logging(state_with_model: State, logger_mock: Logger):
    se = SqueezeExcite(latent_channels=64, min_channels=3)
    se.apply(Event.INIT, state_with_model, logger=logger_mock)
    conv_count = surgery.count_module_instances(state_with_model.model, torch.nn.Conv2d)

    logger_mock.metric_fit.assert_called_once_with({
        'squeeze_excite/num_squeeze_excite_layers': conv_count,
    })


@pytest.mark.run_long
@pytest.mark.timeout(90)
def test_squeeze_excite_trains(mosaic_trainer_hparams: TrainerHparams):
    mosaic_trainer_hparams.algorithms = [SqueezeExciteHparams(latent_channels=32, min_channels=32)]
    train_model(mosaic_trainer_hparams, run_loss_check=True)
