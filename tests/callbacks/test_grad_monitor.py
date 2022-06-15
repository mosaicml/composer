# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import GradMonitor
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.common.datasets import RandomClassificationDataset
from tests.common.models import SimpleModel


@pytest.mark.parametrize('log_layers', [True, False])
def test_grad_monitor(log_layers: bool):
    # Construct the callback
    grad_monitor = GradMonitor(log_layer_grad_norms=log_layers)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger

    # Construct the trainer and train
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=grad_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='1ep',
    )
    trainer.fit()
    num_train_steps = int(trainer.state.timestamp.batch)

    # Count the logged steps
    grad_norm_calls = len(in_memory_logger.data['grad_l2_norm/step'])
    layer_norm_calls = [len(calls) for (k, calls) in in_memory_logger.data.items() if 'layer_grad_l2_norm' in k]

    # expected to log gradient norm once per step (total batch)
    assert grad_norm_calls == num_train_steps
    if log_layers:
        for num_calls in layer_norm_calls:
            assert num_calls == num_train_steps
