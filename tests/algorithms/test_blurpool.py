# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Test the blurpool algorithm.

Primitives are tested in test_blurpool.py
"""
from typing import List, Sequence, Union
from unittest.mock import Mock

import pytest
import torch

from composer.algorithms import BlurPool
from composer.algorithms.blurpool import apply_blurpool
from composer.algorithms.blurpool.blurpool_layers import BlurConv2d, BlurMaxPool2d
from composer.algorithms.warnings import NoEffectWarning
from composer.core import Event, State
from composer.loggers import Logger
from composer.models import ComposerClassifier
from composer.utils import module_surgery


class ConvModel(torch.nn.Module):
    """Convolution Model with layers designed to test different properties of the blurpool algorithm."""

    def __init__(self):
        super().__init__()

        conv_args = {'kernel_size': (3, 3), 'padding': 1}
        self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=8, stride=2, bias=False, **conv_args)  # stride > 1
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=32, stride=2, bias=False,
                                     **conv_args)  # stride > 1 but in_channels < 16
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, stride=1, bias=False, **conv_args)  # stride = 1

        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        self.pool2 = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(64, 48)
        self.linear2 = torch.nn.Linear(48, 10)

    def forward(self, x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Union[torch.Tensor, Sequence[torch.Tensor]]:

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool1(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


@pytest.fixture
def state(minimal_state: State):
    minimal_state.model = ComposerClassifier(ConvModel())
    return minimal_state


@pytest.fixture(params=[
    # replace_conv, replace_pool, blur_first
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
])
def blurpool_instance(request) -> BlurPool:
    replace_conv, replace_pool, blur_first = request.param
    return BlurPool(
        replace_convs=replace_conv,
        replace_maxpools=replace_pool,
        blur_first=blur_first,
    )


def test_blurconv(state: State, blurpool_instance: BlurPool, empty_logger: Logger):
    blurpool_instance.apply(Event.INIT, state, empty_logger)
    assert isinstance(state.model.module, ConvModel)

    if blurpool_instance.replace_convs:
        assert type(state.model.module.conv1) is BlurConv2d
    else:
        assert type(state.model.module.conv1) is torch.nn.Conv2d


def test_maybe_replace_strided_conv_stride(state: State, blurpool_instance: BlurPool, empty_logger: Logger):
    blurpool_instance.apply(Event.INIT, state, empty_logger)
    assert isinstance(state.model.module, ConvModel)

    assert type(state.model.module.conv3) is torch.nn.Conv2d  # stride = 1, should be no replacement


def test_maybe_replace_strided_conv_channels(state: State, blurpool_instance: BlurPool, empty_logger: Logger):
    blurpool_instance.apply(Event.INIT, state, empty_logger)
    assert isinstance(state.model.module, ConvModel)

    assert type(state.model.module.conv2) is torch.nn.Conv2d  # channels < 16, should be no replacement


def test_blurconv_weights_preserved(state: State, blurpool_instance: BlurPool, empty_logger: Logger):
    assert isinstance(state.model.module, ConvModel)

    original_weights = state.model.module.conv1.weight.clone()
    blurpool_instance.apply(Event.INIT, state, empty_logger)

    if isinstance(state.model.module.conv1, BlurConv2d):
        new_weights = state.model.module.conv1.conv.weight
    elif isinstance(state.model.module.conv1, torch.nn.Conv2d):
        new_weights = state.model.module.conv1.weight
    else:
        raise TypeError(f'Layer type {type(state.model.module.conv1)} not expected.')
    assert torch.allclose(original_weights, new_weights)


def test_blurpool(state: State, blurpool_instance: BlurPool, empty_logger: Logger):
    blurpool_instance.apply(Event.INIT, state, empty_logger)
    assert isinstance(state.model.module, ConvModel)

    if blurpool_instance.replace_maxpools:
        assert type(state.model.module.pool1) is BlurMaxPool2d
    else:
        assert type(state.model.module.pool1) is torch.nn.MaxPool2d


def test_blurpool_wrong_event(state: State, blurpool_instance: BlurPool):
    assert blurpool_instance.match(Event.BATCH_START, state) == False


def test_blurpool_correct_event(state: State, blurpool_instance: BlurPool):
    assert blurpool_instance.match(Event.INIT, state) == True


def test_blurpool_algorithm_logging(state: State, blurpool_instance: BlurPool):
    mock_logger = Mock()

    blurpool_instance.apply(Event.INIT, state, mock_logger)

    mock_logger.data_fit.assert_called_once_with({
        'blurpool/num_blurpool_layers': 1 if blurpool_instance.replace_maxpools else 0,
        'blurpool/num_blurconv_layers': 1 if blurpool_instance.replace_convs else 0,
    })


def test_blurpool_noeffectwarning():
    model = torch.nn.Linear(in_features=16, out_features=32)
    with pytest.warns(NoEffectWarning):
        apply_blurpool(model)


def test_blurpool_min_channels():
    model = torch.nn.Conv2d(in_channels=32, out_channels=64, stride=1, kernel_size=(3, 3))
    with pytest.warns(NoEffectWarning):
        apply_blurpool(model, min_channels=64)


def test_blurconv2d_optimizer_params_updated():

    model = ConvModel()

    original_layer = model.conv1
    assert original_layer.stride == (2, 2)  # fail fast if test model changes

    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    apply_blurpool(model, optimizers=optimizer)

    new_layer = model.conv1
    param_list: List[torch.Tensor] = optimizer.param_groups[0]['params']

    # assert old parameters removed
    assert not module_surgery._tensor_in(original_layer.weight, param_list)

    # new params added
    new_conv_layer = new_layer.conv
    assert isinstance(new_conv_layer, torch.nn.Conv2d)
    assert new_conv_layer.weight is not original_layer.weight
    assert module_surgery._tensor_in(new_conv_layer.weight, param_list)
