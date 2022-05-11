# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from torch.cuda import device_count

from composer.callbacks import MemoryMonitorHparams
from composer.loggers import LoggerDestination
from composer.trainer import TrainerHparams
from composer.trainer.devices import DeviceGPU
from composer.trainer.devices.device_hparams import GPUDeviceHparams
from composer.utils import ensure_tuple


def _do_trainer_fit(composer_trainer_hparams: TrainerHparams, testing_with_gpu: bool = False):
    memory_monitor_hparams = MemoryMonitorHparams()
    composer_trainer_hparams.callbacks.append(memory_monitor_hparams)

    max_epochs = 1
    composer_trainer_hparams.max_duration = f"{max_epochs}ep"

    trainer = composer_trainer_hparams.initialize_object()

    # Default model uses CPU
    if testing_with_gpu:
        trainer._device = DeviceGPU()

    log_destination = MagicMock()
    log_destination = cast(LoggerDestination, log_destination)
    trainer.logger.destinations = ensure_tuple(log_destination)
    trainer.fit()

    num_train_steps = composer_trainer_hparams.train_subset_num_batches
    assert isinstance(num_train_steps, int)

    expected_calls = num_train_steps * max_epochs

    return log_destination, expected_calls


@pytest.mark.timeout(60)
def test_memory_monitor_cpu(composer_trainer_hparams: TrainerHparams):
    log_destination, _ = _do_trainer_fit(composer_trainer_hparams, testing_with_gpu=False)

    if torch.cuda.device_count() > 0:
        pytest.skip('Skip CPU memory monitor tests if CUDA is available.')

    memory_monitor_called = False
    for log_call in log_destination.log_data.mock_calls:
        metrics = log_call[1][2]
        if "memory/alloc_requests" in metrics:
            if metrics["memory/alloc_requests"] > 0:
                memory_monitor_called = True
                break
    assert not memory_monitor_called


@pytest.mark.gpu
def test_memory_monitor_gpu(composer_trainer_hparams: TrainerHparams):
    n_cuda_devices = device_count()
    composer_trainer_hparams.device = GPUDeviceHparams()
    if n_cuda_devices > 0:
        log_destination, expected_calls = _do_trainer_fit(composer_trainer_hparams, testing_with_gpu=True)

        num_memory_monitor_calls = 0

        for log_call in log_destination.log_data.mock_calls:
            metrics = log_call[1][2]
            if "memory/alloc_requests" in metrics:
                if metrics["memory/alloc_requests"] > 0:
                    num_memory_monitor_calls += 1
        assert num_memory_monitor_calls == expected_calls
