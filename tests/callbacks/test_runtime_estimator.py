# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import datetime
import time

import pytest
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


@pytest.mark.parametrize('time_unit', ['seconds', 'minutes', 'hours', 'days'])
def test_runtime_estimator(time_unit: str):
    # Construct the callbacks
    skip_batches = 1
    runtime_estimator = RuntimeEstimator(skip_batches=skip_batches, time_unit=time_unit)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger

    simple_model = SimpleModel()
    original_fwd = simple_model.forward

    def new_fwd(x):
        time.sleep(0.02)
        return original_fwd(x)

    simple_model.forward = new_fwd  # type: ignore

    # Construct the trainer and train
    trainer = Trainer(
        model=simple_model,
        callbacks=runtime_estimator,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        eval_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='2ep',
        eval_interval='1ep',
        train_subset_num_batches=5,
        eval_subset_num_batches=5,
    )
    trainer.fit()

    time_remaining_calls = len(in_memory_logger.data['time/remaining_estimate'])
    _assert_no_negative_values(in_memory_logger.data['time/remaining_estimate'])

    expected_calls = int(trainer.state.timestamp.batch) - skip_batches
    assert time_remaining_calls == expected_calls

    ba_2_estimate = in_memory_logger.data['time/remaining_estimate'][1][-1]
    # Should be ~0.2 seconds
    if time_unit == 'seconds':
        assert ba_2_estimate < 1
        assert ba_2_estimate > 0.1
    elif time_unit == 'minutes':
        assert ba_2_estimate < 1 / 60
        assert ba_2_estimate > 0.1 / 60
    elif time_unit == 'hours':
        assert ba_2_estimate < 1 / 60 / 60
        assert ba_2_estimate > 0.1 / 60 / 60
    elif time_unit == 'days':
        assert ba_2_estimate < 1 / 60 / 60 / 24
        assert ba_2_estimate > 0.1 / 60 / 60 / 24
