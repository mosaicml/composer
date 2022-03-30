import copy

import numpy as np
import pytest
import torch

from composer.algorithms import EMAHparams
from composer.algorithms.ema.ema import ema
from composer.core import Event, Time, Timer, TimeUnit
from tests.common import SimpleConvModel, SimpleModel


def validate_ema(model, original_model, ema_model, smoothing):
    model_dict = model.state_dict()
    original_dict = original_model.state_dict()
    ema_dict = ema_model.state_dict()

    for key, param in original_dict.items():
        model_param = model_dict[key].detach()
        new_param = param * smoothing + (1. - smoothing) * model_param
        torch.testing.assert_allclose(ema_dict[key], new_param)


def validate_model(model1, model2):
    model1_dict = model1.state_dict()
    model2_dict = model2.state_dict()
    for key, param in model1_dict.items():
        torch.testing.assert_allclose(param, model2_dict[key])


@pytest.mark.parametrize("smoothing", [0, 0.5, 0.99, 1])
def test_ema(smoothing):
    model = SimpleModel()
    ema_model = SimpleModel()
    original_model = copy.deepcopy(ema_model)
    ema(model=model, ema_model=ema_model, smoothing=smoothing)
    validate_ema(model, original_model, ema_model, smoothing)


# params = [(half_life, update_interval)]
@pytest.mark.parametrize('params', [("10ba", "1ba"), ("1ep", "1ep")])
def test_ema_algorithm(params, minimal_state, empty_logger):

    # Initialize input tensor
    input = torch.rand((32, 5))

    half_life, update_interval = params[0], params[1]
    algorithm = EMAHparams(half_life=half_life, update_interval=update_interval,
                           train_with_ema_weights=False).initialize_object()
    state = minimal_state
    state.model = SimpleConvModel()
    state.batch = (input, torch.Tensor())

    # Start EMA
    algorithm.apply(Event.FIT_START, state, empty_logger)
    # Check if ema correctly calculated smoothing
    half_life = Time.from_timestring(params[0])
    update_interval = Time.from_timestring(params[1])
    smoothing = np.exp(-np.log(2) * (update_interval.value / half_life.value))
    np.testing.assert_allclose(smoothing, algorithm.smoothing)

    # Fake a training update by replacing state.model after ema grabbed it.
    original_model = copy.deepcopy(state.model)
    state.model = SimpleConvModel()
    # Do the EMA update
    state.timer = Timer()
    if half_life.unit == TimeUnit.BATCH:
        state.timer._batch = update_interval
        algorithm.apply(Event.BATCH_END, state, empty_logger)
    elif half_life.unit == TimeUnit.EPOCH:
        state.timer._epoch = update_interval
        algorithm.apply(Event.EPOCH_END, state, empty_logger)
    else:
        raise ValueError(f"Invalid time string for parameter half_life")
    # Check if EMA correctly computed the average.
    validate_ema(state.model, original_model, algorithm.ema_model, algorithm.smoothing)
    # Check if the EMA model is swapped in for testing
    algorithm.apply(Event.EVAL_START, state, empty_logger)
    validate_model(state.model, algorithm.ema_model)
    # Check if the training model is swapped back in for training
    algorithm.apply(Event.EVAL_END, state, empty_logger)
    validate_model(state.model, algorithm.training_model)
