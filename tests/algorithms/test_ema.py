# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pytest
import torch

from composer.algorithms import EMA
from composer.algorithms.ema.ema import EMAParameters, compute_ema
from composer.core import Event, Time, Timestamp, TimeUnit
from tests.common import SimpleConvModel, SimpleTransformerClassifier
from tests.common.models import configure_tiny_bert_hf_model


def validate_ema(model, original_model, ema_model, smoothing):
    model_params, model_buffers = dict(model.named_parameters()), dict(model.named_buffers())
    original_params, original_buffers = dict(original_model.named_parameters()), dict(original_model.named_buffers())
    ema_params, ema_buffers = dict(ema_model.named_parameters()), dict(ema_model.named_buffers())

    for name, param in model_params.items():
        new_param = (original_params[name] * smoothing + (1. - smoothing) * param)
        torch.testing.assert_close(ema_params[name].data, new_param)

    for name, buffer in model_buffers.items():
        new_buffer = (original_buffers[name] * smoothing + (1. - smoothing) * buffer).type(ema_buffers[name].data.dtype)
        torch.testing.assert_close(ema_buffers[name].data, new_buffer)


def validate_model(model1, model2):
    model1_params, model1_buffers = dict(model1.named_parameters()), dict(model1.named_buffers())
    model2_params, model2_buffers = dict(model2.named_parameters()), dict(model2.named_buffers())

    for name, _ in model1_params.items():
        torch.testing.assert_close(model1_params[name].data, model2_params[name].data)

    for name, _ in model1_buffers.items():
        torch.testing.assert_close(model1_buffers[name].data, model2_buffers[name].data)


@pytest.mark.parametrize('smoothing', [0, 0.5, 0.99, 1])
@pytest.mark.parametrize('model_cls', [(SimpleConvModel), (SimpleTransformerClassifier),
                                       (configure_tiny_bert_hf_model)])
def test_ema(smoothing, model_cls):
    model = model_cls()
    ema_model = model_cls()
    original_model = copy.deepcopy(ema_model)
    compute_ema(model=model, ema_model=ema_model, smoothing=smoothing)
    validate_ema(model, original_model, ema_model, smoothing)


# params = [(half_life, update_interval)]
@pytest.mark.parametrize('params', [{
    'half_life': '10ba',
    'update_interval': '1ba'
}, {
    'half_life': '1ep',
    'update_interval': '1ep'
}, {
    'smoothing': 0.999,
    'update_interval': '1ba'
}])
@pytest.mark.parametrize('model_cls', [(SimpleConvModel), (SimpleTransformerClassifier),
                                       (configure_tiny_bert_hf_model)])
def test_ema_algorithm(params, model_cls, minimal_state, empty_logger):

    # Initialize input tensor
    input = torch.rand((32, 5))
    if 'smoothing' in params:
        smoothing, update_interval = params['smoothing'], params['update_interval']
        algorithm = EMA(half_life=None, smoothing=smoothing, update_interval=update_interval)
    else:
        half_life, update_interval = params['half_life'], params['update_interval']
        algorithm = EMA(half_life=half_life, update_interval=update_interval)
    state = minimal_state
    state.model = model_cls()
    state.batch = (input, torch.Tensor())

    # Start EMA
    algorithm.ema_model = EMAParameters(state.model)
    # Check if ema correctly calculated smoothing
    update_interval = Time.from_timestring(params['update_interval'])
    if 'half_life' in params:
        half_life = Time.from_timestring(params['half_life'])
        smoothing = np.exp(-np.log(2) * (update_interval.value / half_life.value))
        np.testing.assert_allclose(np.asarray(smoothing), np.asarray(algorithm.smoothing))

    # Fake a training update by replacing state.model after ema grabbed it.
    original_model = copy.deepcopy(state.model)
    state.model = model_cls()
    training_updated_model = copy.deepcopy(state.model)
    # Do the EMA update
    state.timestamp = Timestamp()
    if update_interval.unit == TimeUnit.BATCH:
        state.timestamp._batch = update_interval
        algorithm.apply(Event.BATCH_END, state, empty_logger)
    elif update_interval.unit == TimeUnit.EPOCH:
        state.timestamp._epoch = update_interval
        algorithm.apply(Event.EPOCH_END, state, empty_logger)
    else:
        raise ValueError(f'Invalid time string for parameter half_life')
    # Check if EMA correctly computed the average.
    validate_ema(state.model, original_model, algorithm.ema_model, algorithm.smoothing)
    ema_updated_model = copy.deepcopy(algorithm.ema_model)
    # Check if the EMA model is swapped in for testing
    algorithm.apply(Event.EVAL_START, state, empty_logger)
    validate_model(state.model, ema_updated_model)
    # Check if the training model is swapped back in for training
    algorithm.apply(Event.EVAL_END, state, empty_logger)
    validate_model(state.model, training_updated_model)
