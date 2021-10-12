from copy import deepcopy

import torch

from composer.algorithms import LayerFreezing
from composer.core.state import State
from composer.core.types import Event, Model, Precision
from composer.loggers import Logger
from composer.utils import ensure_tuple


def _generate_state(epoch: int, max_epochs: int, model: Model):
    state = State(
        epoch=epoch,
        step=epoch,
        train_batch_size=64,
        eval_batch_size=64,
        grad_accum=1,
        max_epochs=max_epochs,
        model=model,
        optimizers=(torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99),),
        precision=Precision.FP32,
    )
    return state


def _check_param_groups(expected_groups, actual_groups):
    assert len(expected_groups) == len(actual_groups), 'Incorrect number of param groups'

    for i, expected_group in enumerate(expected_groups):
        assert len(expected_group) == len(actual_groups[i]), \
            f'Group {i} has the wrong number of parameters'

        for j, expected_params in enumerate(expected_group['params']):
            torch.testing.assert_equal(actual_groups[i]['params'][j], expected_params)


def test_freeze_layers_no_freeze(simple_conv_model: Model, noop_dummy_logger: Logger):
    state = _generate_state(epoch=10, max_epochs=100, model=simple_conv_model)

    first_optimizer = ensure_tuple(state.optimizers)[0]
    assert first_optimizer is not None

    expected_param_groups = deepcopy(first_optimizer.param_groups)
    freezing = LayerFreezing(freeze_start=0.5, freeze_level=1.0)
    freezing.apply(event=Event.EPOCH_END, state=state, logger=noop_dummy_logger)
    updated_param_groups = first_optimizer.param_groups

    _check_param_groups(expected_param_groups, updated_param_groups)


def test_freeze_layers_with_freeze(simple_conv_model: Model, noop_dummy_logger: Logger):
    state = _generate_state(epoch=80, max_epochs=100, model=simple_conv_model)

    first_optimizer = ensure_tuple(state.optimizers)[0]
    assert first_optimizer is not None

    expected_param_groups = deepcopy(first_optimizer.param_groups)
    # The first group should be removed due to freezing
    expected_param_groups[0]['params'] = expected_param_groups[0]['params'][1:]
    freezing = LayerFreezing(freeze_start=0.05, freeze_level=1.0)
    freezing.apply(event=Event.EPOCH_END, state=state, logger=noop_dummy_logger)
    updated_param_groups = first_optimizer.param_groups

    _check_param_groups(expected_param_groups, updated_param_groups)
