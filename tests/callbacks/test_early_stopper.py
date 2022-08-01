# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
from torch.utils.data import DataLoader

from composer import Trainer
from composer.callbacks.early_stopper import EarlyStopper
from composer.core.time import Time, TimeUnit
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU
from tests.common import RandomClassificationDataset, SimpleModel, device
from tests.metrics import MetricSetterCallback


@device('cpu', 'gpu')
@pytest.mark.parametrize('metric_sequence', [[0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3], [0.1, 0.2]])
@pytest.mark.parametrize('unit', [TimeUnit.EPOCH, TimeUnit.BATCH])
def test_early_stopper(metric_sequence: List[float], unit: TimeUnit, device: str):

    if unit == TimeUnit.EPOCH:
        dataloader_label = 'eval'
    else:
        dataloader_label = 'train'

    test_device = DeviceGPU() if device == 'gpu' else DeviceCPU()

    early_stopper = EarlyStopper('Accuracy', dataloader_label, patience=Time(3, unit))

    test_metric_setter = MetricSetterCallback('Accuracy', dataloader_label, metric_sequence, unit, test_device)

    trainer = Trainer(
        model=SimpleModel(num_features=5),
        train_dataloader=DataLoader(
            RandomClassificationDataset(shape=(5, 1, 1)),
            batch_size=4,
        ),
        eval_dataloader=DataLoader(
            RandomClassificationDataset(shape=(5, 1, 1)),
            batch_size=4,
        ),
        max_duration='30ep',
        callbacks=[test_metric_setter, early_stopper],
    )

    trainer.fit()

    assert trainer.state.timestamp.get(unit).value == len(metric_sequence) + int(early_stopper.patience)
