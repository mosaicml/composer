# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

from torch.utils.data import DataLoader

from composer import Callback, Event, State, Trainer
from composer.loggers import FileLogger, Logger, LoggerDestination
from composer.utils.collect_env import disable_env_report
from tests.common.datasets import RandomClassificationDataset
from tests.common.models import SimpleModel


class FileUploaderTracker(LoggerDestination):

    def __init__(self) -> None:
        self.uploaded_files = []

    def upload_file(self, state: State, remote_file_name: str, file_path: pathlib.Path, *, overwrite: bool):
        del state, overwrite  # unused
        self.uploaded_files.append((remote_file_name, file_path))


def test_file_logger(dummy_state: State, tmp_path: pathlib.Path):
    log_file_name = os.path.join(tmp_path, 'output.log')
    log_destination = FileLogger(
        filename=log_file_name,
        remote_file_name='{run_name}/rank{rank}.log',
        buffer_size=1,
        flush_interval=1,
    )
    file_tracker_destination = FileUploaderTracker()
    logger = Logger(dummy_state, destinations=[log_destination, file_tracker_destination])
    log_destination.run_event(Event.INIT, dummy_state, logger)
    logger.log_hyperparameters({'foo': 3})
    logger.log_metrics({'loss': 2}, step=1)
    log_destination.run_event(Event.BATCH_END, dummy_state, logger)
    log_destination.close(dummy_state, logger)
    with open(log_file_name, 'r') as f:
        assert f.readlines() == [
            '[hyperparameter]: foo: 3 \n',
            '[metric][batch=1]: loss: 2 \n',
        ]


def test_file_logger_capture_stdout_stderr(dummy_state: State, tmp_path: pathlib.Path):
    log_file_name = os.path.join(tmp_path, 'output.log')
    log_destination = FileLogger(filename=log_file_name,
                                 buffer_size=1,
                                 flush_interval=1,
                                 capture_stderr=True,
                                 capture_stdout=True)
    # capturing should start immediately
    print('Hello, stdout!\nExtra Line')
    print('Hello, stderr!\nExtra Line2', file=sys.stderr)
    logger = Logger(dummy_state, destinations=[log_destination])
    log_destination.run_event(Event.INIT, dummy_state, logger)
    log_destination.run_event(Event.EPOCH_START, dummy_state, logger)
    log_destination.run_event(Event.BATCH_START, dummy_state, logger)
    log_destination.run_event(Event.BATCH_END, dummy_state, logger)
    log_destination.close(dummy_state, logger)
    with open(log_file_name, 'r') as f:
        assert f.readlines() == [
            '[stdout]: Hello, stdout!\n',
            '[stdout]: Extra Line\n',
            '[stderr]: Hello, stderr!\n',
            '[stderr]: Extra Line2\n',
        ]


class ExceptionRaisingCallback(Callback):

    def fit_start(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        raise RuntimeError('My Exception!')


def test_exceptions_are_printed(tmp_path: pathlib.Path):
    # Test that exceptions are printed to stderr, which is captured by the file logger
    # The file logger stops capturing stdout/stderr when it is closed
    # Here, we construct a trainer that raises an exception on Event.FIT_START
    # and assert that the exception is written to the logfile
    exception_raising_callback = ExceptionRaisingCallback()
    logfile_name = str(tmp_path / 'logfile.txt')
    file_logger = FileLogger(filename=logfile_name, capture_stderr=True)
    dataloader = DataLoader(RandomClassificationDataset())
    model = SimpleModel()
    trainer = Trainer(model=model,
                      train_dataloader=dataloader,
                      max_duration=1,
                      callbacks=[exception_raising_callback],
                      loggers=[file_logger])
    disable_env_report()  # Printing the full report in this test can cause timeouts
    # manually calling `sys.excepthook` for the exception, as it is impossible to write a test
    # that validates unhandled exceptions are logged, since the test validation code would by definition
    # need to handle the exception!
    try:
        trainer.fit()
    except RuntimeError:
        exc_type, exc_value, tb = sys.exc_info()
        assert exc_type is not None
        assert exc_value is not None
        assert tb is not None
        sys.excepthook(exc_type, exc_value, tb)

    trainer.close()

    with open(logfile_name, 'r') as f:
        log_lines = f.readlines()
        assert '[stderr]: RuntimeError: My Exception!\n' == log_lines[-1]

    # Since the trainer was closed, future prints should not appear in the file logger
    print('SHOULD NOT BE CAPTURED')
    with open(logfile_name, 'r') as f:
        logfile = f.read()
        assert 'SHOULD NOT BE CAPTURED' not in logfile
