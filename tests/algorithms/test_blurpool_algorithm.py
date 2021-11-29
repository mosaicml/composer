# Copyright 2021 MosaicML. All Rights Reserved.

"""
Test the blurpool algorithm. Primitives are tested in test_blurpool.py
"""
import itertools
from unittest.mock import MagicMock

import pytest
import torch

from composer.algorithms import BlurPool, BlurPoolHparams
from composer.algorithms.blurpool.blurpool_layers import BlurConv2d, BlurMaxPool2d
from composer.core import Event, State
from composer.core.types import DataLoader, Model, Precision
from tests.fixtures.models import SimpleConvModel


@pytest.fixture
def state(simple_conv_model: Model, dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader):
    state = State(
        epoch=50,
        step=50,
        train_batch_size=100,
        eval_batch_size=100,
        grad_accum=1,
        max_epochs=100,
        model=simple_conv_model,
        precision=Precision.FP32,
        train_dataloader=dummy_train_dataloader,
        eval_dataloader=dummy_val_dataloader,
    )
    return state


@pytest.fixture(params=itertools.product([True, False], [True, False], [True, False]))
def blurpool_instance(request) -> BlurPool:
    replace_conv, replace_pool, blur_first = request.param
    blurpool_hparams = BlurPoolHparams(
        replace_convs=replace_conv,
        replace_maxpools=replace_pool,
        blur_first=blur_first,
    )
    return blurpool_hparams.initialize_object()


@pytest.fixture
def dummy_logger():
    return MagicMock()


def test_blurconv(state, blurpool_instance, dummy_logger):
    blurpool_instance.apply(Event.INIT, state, dummy_logger)
    assert isinstance(state.model.module, SimpleConvModel)

    if blurpool_instance.hparams.replace_convs:
        assert type(state.model.module.conv1) is BlurConv2d
    else:
        assert type(state.model.module.conv1) is torch.nn.Conv2d


def test_maybe_replace_strided_conv_stride(state, blurpool_instance, dummy_logger):
    blurpool_instance.apply(Event.INIT, state, dummy_logger)
    assert isinstance(state.model.module, SimpleConvModel)

    assert type(state.model.module.conv3) is torch.nn.Conv2d  # stride = 1, should be no replacement


def test_maybe_replace_strided_conv_channels(state, blurpool_instance, dummy_logger):
    blurpool_instance.apply(Event.INIT, state, dummy_logger)
    assert isinstance(state.model.module, SimpleConvModel)

    assert type(state.model.module.conv2) is torch.nn.Conv2d  # channels < 16, should be no replacement


def test_blurconv_weights_preserved(state, blurpool_instance, dummy_logger):
    assert isinstance(state.model.module, SimpleConvModel)

    original_weights = state.model.module.conv1.weight.clone()
    blurpool_instance.apply(Event.INIT, state, dummy_logger)

    if isinstance(state.model.module.conv1, BlurConv2d):
        new_weights = state.model.module.conv1.conv.weight
    elif isinstance(state.model.module.conv1, torch.nn.Conv2d):
        new_weights = state.model.module.conv1.weight
    else:
        raise TypeError(f'Layer type {type(state.model.module.conv1)} not expected.')
    assert torch.allclose(original_weights, new_weights)


def test_blurpool(state, blurpool_instance, dummy_logger):
    blurpool_instance.apply(Event.INIT, state, dummy_logger)
    assert isinstance(state.model.module, SimpleConvModel)

    if blurpool_instance.hparams.replace_maxpools:
        assert type(state.model.module.pool1) is BlurMaxPool2d
    else:
        assert type(state.model.module.pool1) is torch.nn.MaxPool2d


def test_blurpool_wrong_event(state, blurpool_instance):
    assert blurpool_instance.match(Event.BATCH_START, state) == False


def test_blurpool_correct_event(state, blurpool_instance):
    assert blurpool_instance.match(Event.INIT, state) == True


def test_blurpool_algorithm_logging(state, blurpool_instance, dummy_logger):
    blurpool_instance.apply(Event.INIT, state, dummy_logger)

    dummy_logger.metric_fit.assert_called_once_with({
        'blurpool/num_blurpool_layers': 1 if blurpool_instance.hparams.replace_maxpools else 0,
        'blurpool/num_blurconv_layers': 1 if blurpool_instance.hparams.replace_convs else 0,
    })
