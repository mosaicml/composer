# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.callbacks import OptimizerMonitor
from composer.loggers import InMemoryLogger
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from composer.utils import dist
from tests.common import device, world_size
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
        assert 'cosine/moment_grad/module.2.weight' in in_memory_logger.data.keys()
        assert 'l2_norm/second_moment_sqrt/module.2.weight' in in_memory_logger.data.keys()
        assert 'l2_norm/update/module.2.weight' in in_memory_logger.data.keys()
        assert 'cosine/update_grad/module.2.weight' in in_memory_logger.data.keys()

    # Expected to log gradient norm once per step (total batch)
    assert grad_norm_calls == num_train_steps
    for num_calls in layer_norm_calls:
        assert num_calls == num_train_steps


@device('gpu')
@world_size(1, 2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_optimizer_monitor(device, world_size):
    # Construct the callback
    grad_monitor = OptimizerMonitor(log_optimizer_metrics=True)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    model = SimpleModel()
    dataset = RandomClassificationDataset()
    # Construct the trainer and train
    trainer = Trainer(model=model,
                      callbacks=grad_monitor,
                      loggers=in_memory_logger,
                      train_dataloader=DataLoader(dataset, sampler=dist.get_sampler(dataset)),
                      optimizers=DecoupledAdamW(model.parameters()),
                      max_duration='3ba',
                      fsdp_config={
                          'sharding_strategy': 'FULL_SHARD',
                          'min_params': 10,
                          'cpu_offload': False,
                          'mixed_precision': 'PURE',
                          'backward_prefetch': 'BACKWARD_PRE',
                          'activation_checkpointing': False,
                          'activation_ocpu_offload': False,
                          'verbose': False
                      })
    trainer.fit()
    num_train_steps = int(trainer.state.timestamp.batch)

    # Count the logged steps
    grad_norm_calls = len(in_memory_logger.data['l2_norm/grad/global'])
    layer_norm_calls = [len(calls) for (k, calls) in in_memory_logger.data.items() if 'l2_norm/grad' in k]
    test_keys = [
        'l2_norm/grad/module._fsdp_wrapped_module._fpw_module.4._fsdp_wrapped_module.flat_param',
        'l2_norm/moment/module._fsdp_wrapped_module._fpw_module.4._fsdp_wrapped_module.flat_param',
        'cosine/moment_grad/module._fsdp_wrapped_module._fpw_module.4._fsdp_wrapped_module.flat_param',
        'l2_norm/second_moment_sqrt/module._fsdp_wrapped_module._fpw_module.4._fsdp_wrapped_module.flat_param',
        'l2_norm/update/module._fsdp_wrapped_module._fpw_module.4._fsdp_wrapped_module.flat_param',
        'cosine/update_grad/module._fsdp_wrapped_module._fpw_module.4._fsdp_wrapped_module.flat_param',
    ]
    for key in test_keys:
        assert key in in_memory_logger.data.keys()

    # Expected to log gradient norm once per step (total batch)
    assert grad_norm_calls == num_train_steps
    for num_calls in layer_norm_calls:
        assert num_calls == num_train_steps
