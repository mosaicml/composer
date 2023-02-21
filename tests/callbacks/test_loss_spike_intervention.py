# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import LossSpikeIntervention
from composer.loggers import InMemoryLogger
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from tests.common import device, world_size
from tests.common.datasets import RandomClassificationDataset
from tests.common.models import SimpleModel


def test_loss_spike_intervention():
    # Construct the callback
    grad_monitor = LossSpikeIntervention(metric='l2_norm/moment',
                                         window_moving_average=25,
                                         increase_factor=5,
                                         increase_lookback=500,
                                         plateau_min_duration=100,
                                         end_spike_factor=1.10,
                                         unfreeze_policy={
                                            "timeout": 2,
                                         })
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    model = SimpleModel()
    # Construct the trainer and train
    trainer = Trainer(
        model=model,
        callbacks=grad_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        optimizers=DecoupledAdamW(model.parameters()),
        max_duration='10ba',
    )
    trainer.fit()
    num_train_steps = int(trainer.state.timestamp.batch)

    # Count the logged steps
    grad_norm_calls = len(in_memory_logger.data['l2_norm/grad/global'])
    layer_norm_calls = [len(calls) for (k, calls) in in_memory_logger.data.items() if 'l2_norm/grad' in k]
    assert 'l2_norm/grad/module.2.weight' in in_memory_logger.data.keys()
    if True:
        assert 'l2_norm/moment/module.2.weight' in in_memory_logger.data.keys()
        assert 'cosine/moment_grad/module.2.weight' in in_memory_logger.data.keys()
        assert 'l2_norm/second_moment_sqrt/module.2.weight' in in_memory_logger.data.keys()
        assert 'l2_norm/update/module.2.weight' in in_memory_logger.data.keys()
        assert 'cosine/update_grad/module.2.weight' in in_memory_logger.data.keys()
        assert 'percentage_nonzero/second_moment/module.2.weight' in in_memory_logger.data.keys()
        assert 'layerwise_lr_scaling/module.2.weight' in in_memory_logger.data.keys()
    assert in_memory_logger.data['num_frozen_layers'][0][1] == 0
    # expected to log gradient norm once per step (total batch)
    assert grad_norm_calls == num_train_steps
    for num_calls in layer_norm_calls:
        assert num_calls == num_train_steps
