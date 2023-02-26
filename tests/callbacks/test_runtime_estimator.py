# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import datetime

from torch.utils.data import DataLoader

from composer.callbacks import RuntimeEstimator
from composer.core import Time
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


def _assert_no_negative_values(logged_values):
    for timestamp, v in logged_values:
        del timestamp  # unused
        if isinstance(v, Time):
            assert int(v) >= 0
        elif isinstance(v, datetime.timedelta):
            assert v.total_seconds() >= 0
        else:
            assert v >= 0


def test_runtime_estimator():
    # Construct the callbacks
    skip_batches = 1
    runtime_estimator = RuntimeEstimator(skip_batches=skip_batches)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger

    # Construct the trainer and train
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=runtime_estimator,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        eval_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='2ep',
        eval_interval='1ep',
        train_subset_num_batches=10,
        eval_subset_num_batches=10,
    )
    trainer.fit()

    wall_clock_remaining_calls = len(in_memory_logger.data['wall_clock/remaining_estimate'])
    _assert_no_negative_values(in_memory_logger.data['wall_clock/remaining_estimate'])

    expected_calls = int(trainer.state.timestamp.batch) - skip_batches
    assert wall_clock_remaining_calls == expected_calls
