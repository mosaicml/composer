# Copyright 2021 MosaicML. All Rights Reserved.

from typing import cast

import numpy as np
import pytest
import torch
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.collections import MetricCollection

from composer.algorithms import ChannelsLastHparams
from composer.algorithms.channels_last import apply_channels_last
from composer.core.event import Event
from composer.core.state import State
from composer.core.types import DataLoader, Evaluator, Model, Precision, Tensor


def _has_singleton_dimension(tensor: Tensor) -> bool:
    return any(s == 1 for s in tensor.shape)


def _infer_memory_format(tensor: Tensor) -> str:
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
def state(simple_conv_model: Model, dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader):
    metric_coll = MetricCollection([Accuracy()])
    evaluators = [Evaluator(label="dummy_label", dataloader=dummy_val_dataloader, metrics=metric_coll)]
    return State(
        model=simple_conv_model,
        precision=Precision.FP32,
        grad_accum=1,
        max_duration="10ep",
        train_dataloader=dummy_train_dataloader,
        evaluators=evaluators,
    )


def test_channels_last_functional(simple_conv_model: Model):
    model = cast(torch.nn.Module, simple_conv_model.module)
    conv = cast(torch.nn.Conv2d, model.conv1)
    assert _infer_memory_format(conv.weight) == 'nchw'
    apply_channels_last(simple_conv_model)
    assert _infer_memory_format(conv.weight) == 'nhwc'


def test_channels_last_algorithm(state, dummy_logger):
    channels_last = ChannelsLastHparams().initialize_object()

    assert state.model is not None
    assert _infer_memory_format(state.model.module.conv1.weight) == 'nchw'
    channels_last.apply(Event.TRAINING_START, state, dummy_logger)
    assert _infer_memory_format(state.model.module.conv1.weight) == 'nhwc'


"""
Test helper utility _infer_memory_format
"""


@pytest.fixture(params=[True, False])
def tensor(request) -> Tensor:
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
