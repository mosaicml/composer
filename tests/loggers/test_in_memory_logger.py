# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np

from composer.core import State, Time, Timestamp
from composer.loggers import InMemoryLogger, Logger, LogLevel


def test_in_memory_logger(dummy_state: State):
    in_memory_logger = InMemoryLogger(LogLevel.EPOCH)
    logger = Logger(dummy_state, destinations=[in_memory_logger])
    logger.data_batch({'batch': 'should_be_ignored'})
    logger.data_epoch({'epoch': 'should_be_recorded'})
    dummy_state.timestamp = dummy_state.timestamp.to_next_batch(samples=1, tokens=1)
    logger.data_epoch({'epoch': 'should_be_recorded_and_override'})

    # no batch events should be logged, since the level is epoch
    assert 'batch' not in in_memory_logger.data
    assert len(in_memory_logger.data['epoch']) == 2

    # `in_memory_logger.data` should contain everything
    timestamp, _, data = in_memory_logger.data['epoch'][0]
    assert timestamp.batch == 0
    assert data == 'should_be_recorded'
    timestamp, _, data = in_memory_logger.data['epoch'][1]
    assert timestamp.batch == 1
    assert data == 'should_be_recorded_and_override'

    # the most recent values should have just the last call to epoch
    assert in_memory_logger.most_recent_values['epoch'] == 'should_be_recorded_and_override'
    assert in_memory_logger.most_recent_timestamps['epoch'].batch == 1


def test_in_memory_logger_get_timeseries():
    in_memory_logger = InMemoryLogger(LogLevel.BATCH)
    data = {'accuracy/val': [], 'batch': [], 'batch_in_epoch': []}
    for i in range(10):
        batch = i
        batch_in_epoch = i % 3
        timestamp = Timestamp(
            epoch=Time(0, 'ep'),
            batch=Time(batch, 'ba'),
            batch_in_epoch=Time(batch_in_epoch, 'ba'),
            sample=Time(0, 'sp'),
            sample_in_epoch=Time(0, 'sp'),
            token=Time(0, 'tok'),
            token_in_epoch=Time(0, 'tok'),
        )
        state = MagicMock()
        state.timestamp = timestamp
        datapoint = i / 3
        in_memory_logger.log_data(state=state, log_level=LogLevel.BATCH, data={'accuracy/val': datapoint})
        data['accuracy/val'].append(datapoint)
        data['batch'].append(batch)
        data['batch_in_epoch'].append(batch_in_epoch)

    timeseries = in_memory_logger.get_timeseries('accuracy/val')
    for k, v in data.items():
        assert np.all(timeseries[k] == np.array(v))
