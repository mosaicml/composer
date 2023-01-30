# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import OptimizerMonitor
from composer.loggers import InMemoryLogger
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from tests.common.datasets import RandomClassificationDataset
from tests.common.models import SimpleModel


@pytest.mark.parametrize('log_optimizer_metrics', [True, False])
def test_optimizer_monitor(log_optimizer_metrics: bool):
    # Construct the callback
    grad_monitor = OptimizerMonitor(log_optimizer_metrics=log_optimizer_metrics)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    model = SimpleModel()
    # Construct the trainer and train
    trainer = Trainer(
        model=model,
        callbacks=grad_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        optimizers=DecoupledAdamW(model.parameters()),
        max_duration='3ba',
    )
    trainer.fit()
    num_train_steps = int(trainer.state.timestamp.batch)

    # Count the logged steps
    grad_norm_calls = len(in_memory_logger.data['l2_norm/grad/global'])
    layer_norm_calls = [len(calls) for (k, calls) in in_memory_logger.data.items() if 'l2_norm/grad' in k]
    assert 'l2_norm/grad/module.2.weight' in in_memory_logger.data.keys()

    if log_optimizer_metrics:
        assert 'l2_norm/moment/module.2.weight' in in_memory_logger.data.keys()
        assert 'l2_norm_ratio/moment_grad/module.2.weight' in in_memory_logger.data.keys()
        assert 'cosine/moment_grad/module.2.weight' in in_memory_logger.data.keys()
        assert 'l2_norm/second_moment_sqrt/module.2.weight' in in_memory_logger.data.keys()
        assert 'l2_norm/update/module.2.weight' in in_memory_logger.data.keys()
        assert 'cosine/update_grad/module.2.weight' in in_memory_logger.data.keys()
        assert 'l2_norm_ratio/update_param/module.2.weight' in in_memory_logger.data.keys()

    # expected to log gradient norm once per step (total batch)
    assert grad_norm_calls == num_train_steps
    for num_calls in layer_norm_calls:
        assert num_calls == num_train_steps
