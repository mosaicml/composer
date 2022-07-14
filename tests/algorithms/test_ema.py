# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import itertools

import numpy as np
import pytest
import torch

from composer.algorithms import EMA
from composer.algorithms.ema.ema import ShadowModel, compute_ema
from composer.core import Event, Time, Timestamp, TimeUnit
from tests.common import SimpleConvModel, SimpleModel


def validate_ema(model, original_model, ema_model, smoothing):
    model_params = itertools.chain(model.parameters(), model.buffers())
    original_params = itertools.chain(original_model.parameters(), original_model.buffers())
    ema_params = itertools.chain(ema_model.parameters(), ema_model.buffers())

    for model_param, original_param, ema_param in zip(model_params, original_params, ema_params):
        new_param = original_param * smoothing + (1. - smoothing) * model_param
        torch.testing.assert_close(ema_param.data, new_param)


def validate_model(model1, model2):
    model1_params = itertools.chain(model1.parameters(), model1.buffers())
    model2_params = itertools.chain(model2.parameters(), model2.buffers())

    for model1_param, model2_param in zip(model1_params, model2_params):
        torch.testing.assert_close(model1_param.data, model2_param)


@pytest.mark.parametrize('smoothing', [0, 0.5, 0.99, 1])
def test_ema(smoothing):
    model = SimpleModel()
    ema_model = SimpleModel()
    original_model = copy.deepcopy(ema_model)
    compute_ema(model=model, ema_model=ema_model, smoothing=smoothing)
    validate_ema(model, original_model, ema_model, smoothing)


# params = [(half_life, update_interval)]
@pytest.mark.parametrize('params', [('10ba', '1ba'), ('1ep', '1ep')])
def test_ema_algorithm(params, minimal_state, empty_logger):

    # Initialize input tensor
    input = torch.rand((32, 5))

    half_life, update_interval = params[0], params[1]
    algorithm = EMA(half_life=half_life, update_interval=update_interval, train_with_ema_weights=False)
    state = minimal_state
    state.model = SimpleConvModel()
    state.batch = (input, torch.Tensor())

    # Start EMA
    algorithm.ema_model = ShadowModel(state.model)
    algorithm.training_model = ShadowModel(state.model)
    # Check if ema correctly calculated smoothing
    half_life = Time.from_timestring(params[0])
    update_interval = Time.from_timestring(params[1])
    smoothing = np.exp(-np.log(2) * (update_interval.value / half_life.value))
    np.testing.assert_allclose(smoothing, algorithm.smoothing)

    # Fake a training update by replacing state.model after ema grabbed it.
    original_model = copy.deepcopy(state.model)
    state.model = SimpleConvModel()
    # Do the EMA update
    state.timestamp = Timestamp()
    if half_life.unit == TimeUnit.BATCH:
        state.timestamp._batch = update_interval
        algorithm.apply(Event.BATCH_END, state, empty_logger)
    elif half_life.unit == TimeUnit.EPOCH:
        state.timestamp._epoch = update_interval
        algorithm.apply(Event.EPOCH_END, state, empty_logger)
    else:
        raise ValueError(f'Invalid time string for parameter half_life')
    # Check if EMA correctly computed the average.
    validate_ema(state.model, original_model, algorithm.ema_model, algorithm.smoothing)
    # Check if the EMA model is swapped in for testing
    algorithm.apply(Event.EVAL_START, state, empty_logger)
    validate_model(state.model, algorithm.ema_model)
    # Check if the training model is swapped back in for training
    algorithm.apply(Event.EVAL_END, state, empty_logger)
    validate_model(state.model, algorithm.training_model)
    # Check if the ema model can be extracted correctly
    overwrite_model = copy.deepcopy(original_model)
    overwrite_model = algorithm.get_ema_model(overwrite_model)
    validate_model(overwrite_model, algorithm.ema_model)
