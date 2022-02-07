# Copyright 2021 MosaicML. All Rights Reserved.

from copy import deepcopy

import torch
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.collections import MetricCollection

from composer.algorithms import LayerFreezing, LayerFreezingHparams
from composer.core.logging import Logger
from composer.core.state import State
from composer.core.types import DataLoader, Evaluator, Event, Model, Precision
from composer.trainer.trainer_hparams import TrainerHparams
from tests.utils.trainer_fit import train_model


def _generate_state(epoch: int, max_epochs: int, model: Model, train_dataloader: DataLoader,
                    val_dataloader: DataLoader):
    metric_coll = MetricCollection([Accuracy()])
    evaluators = [Evaluator(label="dummy_label", dataloader=val_dataloader, metrics=metric_coll)]
    state = State(
        grad_accum=1,
        max_duration=f"{max_epochs}ep",
        model=model,
        optimizers=(torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99),),
        precision=Precision.FP32,
        train_dataloader=train_dataloader,
        evaluators=evaluators,
    )
    for _ in range(epoch):
        state.timer.on_epoch_complete()
    return state


def _check_param_groups(expected_groups, actual_groups):
    assert len(expected_groups) == len(actual_groups), 'Incorrect number of param groups'

    for i, expected_group in enumerate(expected_groups):
        assert len(expected_group) == len(actual_groups[i]), \
            f'Group {i} has the wrong number of parameters'

        for j, expected_params in enumerate(expected_group['params']):
            assert (actual_groups[i]['params'][j] == expected_params).all()


def test_freeze_layers_no_freeze(simple_conv_model: Model, noop_dummy_logger: Logger,
                                 dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader):
    state = _generate_state(epoch=10,
                            max_epochs=100,
                            model=simple_conv_model,
                            train_dataloader=dummy_train_dataloader,
                            val_dataloader=dummy_val_dataloader)

    first_optimizer = state.optimizers[0]

    expected_param_groups = deepcopy(first_optimizer.param_groups)
    freezing = LayerFreezing(freeze_start=0.5, freeze_level=1.0)
    freezing.apply(event=Event.EPOCH_END, state=state, logger=noop_dummy_logger)
    updated_param_groups = first_optimizer.param_groups

    _check_param_groups(expected_param_groups, updated_param_groups)


def test_freeze_layers_with_freeze(simple_conv_model: Model, noop_dummy_logger: Logger,
                                   dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader):
    state = _generate_state(epoch=80,
                            max_epochs=100,
                            model=simple_conv_model,
                            train_dataloader=dummy_train_dataloader,
                            val_dataloader=dummy_val_dataloader)

    first_optimizer = state.optimizers[0]

    expected_param_groups = deepcopy(first_optimizer.param_groups)
    # The first group should be removed due to freezing
    expected_param_groups[0]['params'] = []
    freezing = LayerFreezing(freeze_start=0.05, freeze_level=1.0)
    freezing.apply(event=Event.EPOCH_END, state=state, logger=noop_dummy_logger)
    updated_param_groups = first_optimizer.param_groups

    _check_param_groups(expected_param_groups, updated_param_groups)


def test_layer_freezing_trains(composer_trainer_hparams: TrainerHparams):
    composer_trainer_hparams.algorithms = [LayerFreezingHparams(freeze_start=.25, freeze_level=1)]
    train_model(composer_trainer_hparams, max_epochs=4)
