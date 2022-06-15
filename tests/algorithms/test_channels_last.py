# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from composer.algorithms.channels_last import apply_channels_last
from composer.algorithms.channels_last.channels_last import ChannelsLast
from composer.core.event import Event
from composer.core.state import State
from composer.loggers import Logger
from tests.common import SimpleConvModel


def _has_singleton_dimension(tensor: torch.Tensor) -> bool:
    return any(s == 1 for s in tensor.shape)


def _infer_memory_format(tensor: torch.Tensor) -> str:
    if _has_singleton_dimension(tensor):
        raise ValueError(f'Tensor of shape {tensor.shape} has singleton dimensions, '
                         'memory format cannot be infered from strides.')
    base_order = list('nchw')  # type: ignore

    strides = tensor.stride()
    if isinstance(strides, tuple) and len(strides) == 4:
        order = np.argsort(strides)
        # smallest stride should be last in format, so reverse order
        memory_format = ''.join([base_order[o] for o in reversed(order)])
        return memory_format
    else:
        raise ValueError(f'Tensor must be 4-dim, got shape {tensor.shape}')


@pytest.fixture()
def state(minimal_state: State):
    minimal_state.model = SimpleConvModel()
    return minimal_state


@pytest.fixture()
def simple_conv_model():
    return SimpleConvModel()


def test_channels_last_functional(simple_conv_model: SimpleConvModel):
    model = simple_conv_model
    conv = model.conv1
    assert _infer_memory_format(conv.weight) == 'nchw'
    apply_channels_last(simple_conv_model)
    assert _infer_memory_format(conv.weight) == 'nhwc'


@pytest.mark.parametrize(
    'device',
    [pytest.param('cpu'), pytest.param('gpu', marks=pytest.mark.gpu)],
)
def test_channels_last_algorithm(state: State, empty_logger: Logger, device: str):
    channels_last = ChannelsLast()
    if device == 'gpu':
        state.model = state.model.cuda()  # move the model to gpu

    assert isinstance(state.model, SimpleConvModel)
    assert _infer_memory_format(state.model.conv1.weight) == 'nchw'
    channels_last.apply(Event.INIT, state, empty_logger)

    assert isinstance(state.model, SimpleConvModel)
    assert _infer_memory_format(state.model.conv1.weight) == 'nhwc'


# Test helper utility _infer_memory_format


@pytest.fixture(params=[True, False])
def tensor(request) -> torch.Tensor:
    strided = request.param
    tensor = torch.randn((16, 32, 32, 64))
    if strided:
        tensor = tensor[::2, ::2, ::2, ::2]
    return tensor


def test_infer_memory_format_nhwc(tensor):
    tensor = tensor.to(memory_format=torch.channels_last)
    assert _infer_memory_format(tensor) == 'nhwc'


def test_infer_memory_format_nchw(tensor):
    tensor = tensor.to(memory_format=torch.contiguous_format)
    assert _infer_memory_format(tensor) == 'nchw'


def test_infer_memory_format_wcnh(tensor):
    tensor = tensor.to(memory_format=torch.contiguous_format)
    tensor = tensor.permute(2, 1, 3, 0)
    assert _infer_memory_format(tensor) == 'wcnh'


def test_infer_memory_format_incorrect_ndims():
    tensor = torch.randn((16, 32, 32))
    with pytest.raises(ValueError):
        _infer_memory_format(tensor)


def test_infer_memory_format_singleton():
    tensor = torch.randn((16, 32, 1, 64))
    with pytest.raises(ValueError):
        _infer_memory_format(tensor)
