# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch

from composer.algorithms import LayerFreezing
from composer.core import Event, Precision, State, Timestamp
from composer.devices import DeviceCPU, DeviceGPU
from composer.loggers import Logger
from tests.common import SimpleConvModel, SimpleTransformerClassifier
from tests.common.models import configure_tiny_bert_hf_model


def _generate_state(request: pytest.FixtureRequest, model_cls, epoch: int, max_epochs: int):
    """Generates a state and fast forwards the timestamp by epochs."""
    model = model_cls()
    device = None
    for item in request.session.items:
        device = DeviceCPU() if item.get_closest_marker('gpu') is None else DeviceGPU()
        break
    assert device != None
    state = State(model=model,
                  rank_zero_seed=0,
                  device=device,
                  run_name='run_name',
                  optimizers=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99),
                  precision=Precision.FP32,
                  dataloader=Mock(__len__=lambda x: 100),
                  dataloader_label='train',
                  grad_accum=1,
                  max_duration=f'{max_epochs}ep')

    # fast forward by epochs
    state.timestamp = Timestamp(epoch=epoch)

    return state


def _assert_param_groups_equal(expected_groups, actual_groups):
    assert len(expected_groups) == len(actual_groups), 'Incorrect number of param groups'

    for i, expected_group in enumerate(expected_groups):
        assert len(expected_group) == len(actual_groups[i]), \
            f'Group {i} has the wrong number of parameters'

        for j, expected_params in enumerate(expected_group['params']):
            assert (actual_groups[i]['params'][j] == expected_params).all()


@pytest.mark.parametrize('model_cls', [SimpleConvModel, SimpleTransformerClassifier, configure_tiny_bert_hf_model])
def test_freeze_layers_no_freeze(model_cls, empty_logger: Logger, request: pytest.FixtureRequest):
    state = _generate_state(request, model_cls, epoch=10, max_epochs=100)

    first_optimizer = state.optimizers[0]
    expected_param_groups = deepcopy(first_optimizer.param_groups)

    freezing = LayerFreezing(freeze_start=0.5, freeze_level=1.0)
    freezing.apply(event=Event.EPOCH_END, state=state, logger=empty_logger)
    updated_param_groups = first_optimizer.param_groups

    _assert_param_groups_equal(expected_param_groups, updated_param_groups)


@pytest.mark.parametrize('model_cls', [SimpleConvModel, SimpleTransformerClassifier, configure_tiny_bert_hf_model])
def test_freeze_layers_with_freeze(model_cls, empty_logger: Logger, request: pytest.FixtureRequest):
    state = _generate_state(request, model_cls, epoch=80, max_epochs=100)

    first_optimizer = state.optimizers[0]
    expected_param_groups = deepcopy(first_optimizer.param_groups)

    freezing = LayerFreezing(freeze_start=0.05, freeze_level=1.0)
    freezing.apply(event=Event.EPOCH_END, state=state, logger=empty_logger)
    updated_param_groups = first_optimizer.param_groups

    # The first group should be removed due to freezing
    expected_param_groups[0]['params'] = []

    _assert_param_groups_equal(expected_param_groups, updated_param_groups)
