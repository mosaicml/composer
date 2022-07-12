# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
from torch.utils.data import DataLoader

from composer import Trainer
from composer.callbacks import ThresholdStopper
from composer.core.time import TimeUnit
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU
from tests.common import RandomClassificationDataset, SimpleModel, device
from tests.metrics import MetricSetterCallback


@device('cpu', 'gpu')
@pytest.mark.parametrize('metric_sequence', [[0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8], [0.6, 0.7]])
@pytest.mark.parametrize('unit', [TimeUnit.EPOCH, TimeUnit.BATCH])
def test_threshold_stopper_eval(metric_sequence: List[float], unit: TimeUnit, device: str):
    metric_threshold = 0.65

    if unit == TimeUnit.EPOCH:
        dataloader_label = 'eval'
        stop_on_batch = False
    else:
        dataloader_label = 'train'
        stop_on_batch = True

    test_device = DeviceGPU() if device == 'gpu' else DeviceCPU()

    tstop = ThresholdStopper('Accuracy', dataloader_label, metric_threshold, comp=None, stop_on_batch=stop_on_batch)

    test_metric_setter = MetricSetterCallback('Accuracy', dataloader_label, metric_sequence, unit, test_device)

    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=DataLoader(RandomClassificationDataset()),
        eval_dataloader=DataLoader(RandomClassificationDataset()),
        train_subset_num_batches=1,
        eval_subset_num_batches=1,
        device=test_device,
        max_duration='30ep',
        callbacks=[test_metric_setter, tstop],
    )

    trainer.fit()

    count_before_threshold = 0
    for metric in metric_sequence:
        if metric_threshold > metric:
            count_before_threshold += 1

    assert trainer.state.timestamp.get(unit).value == count_before_threshold
