# Copyright 2021 MosaicML. All Rights Reserved.

"""
Test the blurpool algorithm. Primitives are tested in test_blurpool.py
"""
import itertools
from typing import List
from unittest.mock import MagicMock

import pytest
import torch

from composer.algorithms import BlurPool, BlurPoolHparams
from composer.algorithms.blurpool import apply_blurpool
from composer.algorithms.blurpool.blurpool_layers import BlurConv2d, BlurMaxPool2d
from composer.core import Event, State, surgery
from composer.core.types import DataLoader, Logger, Model, Precision
from tests.fixtures.models import SimpleConvModel


@pytest.fixture
def state(simple_conv_model: Model, dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader):
    state = State(
        grad_accum=1,
        max_duration="100ep",
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


def test_blurconv(state: State, blurpool_instance: BlurPool, dummy_logger: Logger):
    blurpool_instance.apply(Event.INIT, state, dummy_logger)
    assert isinstance(state.model.module, SimpleConvModel)

    if blurpool_instance.hparams.replace_convs:
        assert type(state.model.module.conv1) is BlurConv2d
    else:
        assert type(state.model.module.conv1) is torch.nn.Conv2d


def test_maybe_replace_strided_conv_stride(state: State, blurpool_instance: BlurPool, dummy_logger: Logger):
    blurpool_instance.apply(Event.INIT, state, dummy_logger)
    assert isinstance(state.model.module, SimpleConvModel)

    assert type(state.model.module.conv3) is torch.nn.Conv2d  # stride = 1, should be no replacement


def test_maybe_replace_strided_conv_channels(state: State, blurpool_instance: BlurPool, dummy_logger: Logger):
    blurpool_instance.apply(Event.INIT, state, dummy_logger)
    assert isinstance(state.model.module, SimpleConvModel)

    assert type(state.model.module.conv2) is torch.nn.Conv2d  # channels < 16, should be no replacement


def test_blurconv_weights_preserved(state: State, blurpool_instance: BlurPool, dummy_logger: Logger):
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


def test_blurpool(state: State, blurpool_instance: BlurPool, dummy_logger: Logger):
    blurpool_instance.apply(Event.INIT, state, dummy_logger)
    assert isinstance(state.model.module, SimpleConvModel)

    if blurpool_instance.hparams.replace_maxpools:
        assert type(state.model.module.pool1) is BlurMaxPool2d
    else:
        assert type(state.model.module.pool1) is torch.nn.MaxPool2d


def test_blurpool_wrong_event(state: State, blurpool_instance: BlurPool):
    assert blurpool_instance.match(Event.BATCH_START, state) == False


def test_blurpool_correct_event(state: State, blurpool_instance: BlurPool):
    assert blurpool_instance.match(Event.INIT, state) == True


def test_blurpool_algorithm_logging(state: State, blurpool_instance: BlurPool, dummy_logger: Logger):
    blurpool_instance.apply(Event.INIT, state, dummy_logger)

    dummy_logger.metric_fit.assert_called_once_with({
        'blurpool/num_blurpool_layers': 1 if blurpool_instance.hparams.replace_maxpools else 0,
        'blurpool/num_blurconv_layers': 1 if blurpool_instance.hparams.replace_convs else 0,
    })


def test_blurconv2d_optimizer_params_updated():
    model = SimpleConvModel()
    orig_conv = model.conv1
    assert orig_conv.stride == (2, 2)  # fail fast if test model changes
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    apply_blurpool(model, optimizers=opt)
    new_conv = model.conv1
    param_list: List[torch.Tensor] = opt.param_groups[0]['params']

    # old params removed
    assert not surgery._tensor_in(orig_conv.weight, param_list)

    # new params added
    new_conv2d = new_conv.conv
    assert isinstance(new_conv2d, torch.nn.Module)
    new_weight = new_conv2d.weight
    assert new_weight is not orig_conv.weight
    assert isinstance(new_weight, torch.Tensor)
    assert surgery._tensor_in(new_weight, param_list)
