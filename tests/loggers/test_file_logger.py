# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib

import pytest

from composer import Event, State
from composer.loggers import FileLoggerHparams, Logger, LogLevel


@pytest.fixture
def log_file_name(tmpdir: pathlib.Path) -> str:
    return os.path.join(tmpdir, "output.log")


@pytest.mark.parametrize("log_level", [LogLevel.EPOCH, LogLevel.BATCH])
@pytest.mark.timeout(10)
def test_file_logger(dummy_state: State, log_level: LogLevel, log_file_name: str):
    log_destination = FileLoggerHparams(
        log_interval=3,
        log_level=log_level,
        filename=log_file_name,
        buffer_size=1,
        flush_interval=1,
    ).initialize_object()
    logger = Logger(dummy_state, destinations=[log_destination])
    log_destination.run_event(Event.INIT, dummy_state, logger)
    log_destination.run_event(Event.EPOCH_START, dummy_state, logger)
    log_destination.run_event(Event.BATCH_START, dummy_state, logger)
    dummy_state.timer.on_batch_complete()
    log_destination.run_event(Event.BATCH_START, dummy_state, logger)
    dummy_state.timer.on_batch_complete()
    log_destination.run_event(Event.BATCH_START, dummy_state, logger)
    dummy_state.timer.on_epoch_complete()
    log_destination.run_event(Event.EPOCH_START, dummy_state, logger)
    logger.data_fit({"metric": "fit"})  # should print
    logger.data_epoch({"metric": "epoch"})  # should print on batch level, since epoch calls are always printed
    logger.data_batch({"metric": "batch"})  # should print on batch level, since we print every 3 steps
    dummy_state.timer.on_epoch_complete()
    log_destination.run_event(Event.EPOCH_START, dummy_state, logger)
    logger.data_epoch({"metric": "epoch1"})  # should print, since we log every 3 epochs
    dummy_state.timer.on_epoch_complete()
    log_destination.run_event(Event.EPOCH_START, dummy_state, logger)
    dummy_state.timer.on_batch_complete()
    log_destination.run_event(Event.BATCH_START, dummy_state, logger)
    log_destination.run_event(Event.BATCH_END, dummy_state, logger)
    logger.data_epoch({"metric": "epoch2"})  # should print on batch level, since epoch calls are always printed
    logger.data_batch({"metric": "batch1"})  # should NOT print
    log_destination.run_event(Event.BATCH_END, dummy_state, logger)
    log_destination.close()
    with open(log_file_name, 'r') as f:
        if log_level == LogLevel.EPOCH:
            assert f.readlines() == [
                '[FIT][step=2]: { "metric": "fit", }\n',
                '[EPOCH][step=2]: { "metric": "epoch1", }\n',
            ]
        else:
            assert log_level == LogLevel.BATCH
            assert f.readlines() == [
                '[FIT][step=2]: { "metric": "fit", }\n',
                '[EPOCH][step=2]: { "metric": "epoch", }\n',
                '[BATCH][step=2]: { "metric": "batch", }\n',
                '[EPOCH][step=2]: { "metric": "epoch1", }\n',
                '[EPOCH][step=3]: { "metric": "epoch2", }\n',
            ]
