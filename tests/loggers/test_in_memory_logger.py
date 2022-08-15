# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np

from composer.core import State, Time, Timestamp
from composer.loggers import InMemoryLogger, Logger


def test_in_memory_logger(dummy_state: State):
    in_memory_logger = InMemoryLogger()
    logger = Logger(dummy_state, destinations=[in_memory_logger])
    in_memory_logger.init(dummy_state, logger)
    logger.log_metrics({'epoch': 'should_be_recorded'})
    dummy_state.timestamp = dummy_state.timestamp.to_next_batch(samples=1, tokens=1)
    logger.log_metrics({'epoch': 'should_be_recorded_and_override'})


    # `in_memory_logger.data` should contain everything
    timestamp, data = in_memory_logger.data['epoch'][0]
    assert timestamp.batch == 0
    assert data == 'should_be_recorded'
    timestamp, data = in_memory_logger.data['epoch'][1]
    assert timestamp.batch == 1
    assert data == 'should_be_recorded_and_override'

    # the most recent values should have just the last call to epoch
    assert in_memory_logger.most_recent_values['epoch'] == 'should_be_recorded_and_override'
    assert in_memory_logger.most_recent_timestamps['epoch'].batch == 1


def test_in_memory_logger_get_timeseries():
    in_memory_logger = InMemoryLogger()
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
        logger = MagicMock()
        state.timestamp = timestamp
        datapoint = i / 3
        in_memory_logger.init(state, logger)
        in_memory_logger.log_metrics({'accuracy/val': datapoint}, step=state.timestamp.batch.value)
        data['accuracy/val'].append(datapoint)
        data['batch'].append(batch)
        data['batch_in_epoch'].append(batch_in_epoch)

    timeseries = in_memory_logger.get_timeseries('accuracy/val')
    for k, v in data.items():
        assert np.all(timeseries[k] == np.array(v))
