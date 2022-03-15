# Copyright 2021 MosaicML. All Rights Reserved.

from copy import deepcopy
from unittest.mock import Mock

import torch

from composer.algorithms import LayerFreezing, LayerFreezingHparams
from composer.core import Event, State
from composer.core.types import Precision
from composer.loggers import Logger
from tests.common import SimpleConvModel


def _generate_state(epoch: int, max_epochs: int):
    """Generates a state and fast forwards the timer by epochs."""
    model = SimpleConvModel()

    state = State(model=model,
                  rank_zero_seed=0,
                  optimizers=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99),
                  precision=Precision.FP32,
                  grad_accum=1,
                  train_dataloader=Mock(__len__=lambda x: 100),
                  evaluators=Mock(),
                  max_duration=f'{max_epochs}ep')

    # fast forward by epochs
    for _ in range(epoch):
        state.timer.on_epoch_complete()

    return state


def _assert_param_groups_equal(expected_groups, actual_groups):
    assert len(expected_groups) == len(actual_groups), 'Incorrect number of param groups'

    for i, expected_group in enumerate(expected_groups):
        assert len(expected_group) == len(actual_groups[i]), \
            f'Group {i} has the wrong number of parameters'

        for j, expected_params in enumerate(expected_group['params']):
            assert (actual_groups[i]['params'][j] == expected_params).all()


def test_freeze_layers_no_freeze(empty_logger: Logger):
    state = _generate_state(epoch=10, max_epochs=100)

    first_optimizer = state.optimizers[0]
    expected_param_groups = deepcopy(first_optimizer.param_groups)

    freezing = LayerFreezing(freeze_start=0.5, freeze_level=1.0)
    freezing.apply(event=Event.EPOCH_END, state=state, logger=empty_logger)
    updated_param_groups = first_optimizer.param_groups

    _assert_param_groups_equal(expected_param_groups, updated_param_groups)


def test_freeze_layers_with_freeze(empty_logger: Logger):
    state = _generate_state(epoch=80, max_epochs=100)

    first_optimizer = state.optimizers[0]
    expected_param_groups = deepcopy(first_optimizer.param_groups)

    freezing = LayerFreezing(freeze_start=0.05, freeze_level=1.0)
    freezing.apply(event=Event.EPOCH_END, state=state, logger=empty_logger)
    updated_param_groups = first_optimizer.param_groups

    # The first group should be removed due to freezing
    expected_param_groups[0]['params'] = []

    _assert_param_groups_equal(expected_param_groups, updated_param_groups)


def test_layer_freezing_hparams():
    hparams = LayerFreezingHparams(freeze_start=0.05, freeze_level=1.0)
    algorithm = hparams.initialize_object()

    assert isinstance(algorithm, LayerFreezing)
