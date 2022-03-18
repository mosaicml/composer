# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib
import sys

import pytest

from composer import Event, State
from composer.loggers import FileLoggerHparams, Logger, LogLevel


@pytest.mark.parametrize("log_level", [LogLevel.EPOCH, LogLevel.BATCH])
@pytest.mark.timeout(10)
def test_file_logger(dummy_state: State, log_level: LogLevel, tmpdir: pathlib.Path):
    log_file_name = os.path.join(tmpdir, "output.log")
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
                '[FIT][batch=2]: { "metric": "fit", }\n',
                '[EPOCH][batch=2]: { "metric": "epoch1", }\n',
            ]
        else:
            assert log_level == LogLevel.BATCH
            assert f.readlines() == [
                '[FIT][batch=2]: { "metric": "fit", }\n',
                '[EPOCH][batch=2]: { "metric": "epoch", }\n',
                '[BATCH][batch=2]: { "metric": "batch", }\n',
                '[EPOCH][batch=2]: { "metric": "epoch1", }\n',
                '[EPOCH][batch=3]: { "metric": "epoch2", }\n',
            ]


def test_file_logger_capture_stdout_stderr(dummy_state: State, tmpdir: pathlib.Path):
    log_file_name = os.path.join(tmpdir, "output.log")
    log_destination = FileLoggerHparams(filename=log_file_name,
                                        buffer_size=1,
                                        flush_interval=1,
                                        capture_stderr=True,
                                        capture_stdout=True).initialize_object()
    # capturing should start immediately
    print("Hello, stdout!\nExtra Line")
    print("Hello, stderr!\nExtra Line2", file=sys.stderr)
    logger = Logger(dummy_state, destinations=[log_destination])
    log_destination.run_event(Event.INIT, dummy_state, logger)
    log_destination.run_event(Event.EPOCH_START, dummy_state, logger)
    log_destination.run_event(Event.BATCH_START, dummy_state, logger)
    log_destination.run_event(Event.BATCH_END, dummy_state, logger)
    log_destination.close()
    with open(log_file_name, 'r') as f:
        assert f.readlines() == [
            '[stdout]: Hello, stdout!\n',
            '[stdout]: Extra Line\n',
            '[stderr]: Hello, stderr!\n',
            '[stderr]: Extra Line2\n',
        ]
