# Copyright 2021 MosaicML. All Rights Reserved.

import os

import pytest
import torch.distributed as dist
from _pytest.monkeypatch import MonkeyPatch

from composer.core.logging import Logger, LogLevel
from composer.core.state import State
from composer.loggers.file_logger import FileLoggerBackend
from composer.loggers.logger_hparams import FileLoggerBackendHparams


@pytest.fixture
def log_file_name(ddp_tmpdir: str) -> str:
    return os.path.join(ddp_tmpdir, "output.log")


@pytest.fixture
def log_destination(log_file_name: str) -> FileLoggerBackend:
    return FileLoggerBackendHparams(
        every_n_batches=3,
        every_n_epochs=2,
        log_level=LogLevel.BATCH,
        filename=log_file_name,
        buffer_size=1,
        flush_every_n_batches=1,
    ).initialize_object()


def test_file_logger(dummy_state: State, log_destination: FileLoggerBackend, monkeypatch: MonkeyPatch,
                     log_file_name: str):
    dummy_state.step = 2
    dummy_state.epoch = 2
    logger = Logger(dummy_state, backends=[log_destination])
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    log_destination.training_start(dummy_state, logger)
    logger.metric_fit({"metric": "fit"})  # should print
    logger.metric_epoch({"metric": "epoch"})  # should print
    logger.metric_batch({"metric": "batch"})  # should print
    logger.metric_verbose({"metric": "verbose"})  # should NOT print, since we're on the BATCH log level
    dummy_state.epoch = 3
    logger.metric_epoch({"metric": "epoch1"})  # should NOT print, since we print every 2 epochs
    dummy_state.epoch = 4
    dummy_state.step = 3
    log_destination.batch_end(dummy_state, logger)
    logger.metric_epoch({"metric": "epoch2"})  # should print
    logger.metric_batch({"metric": "batch1"})  # should NOT print, since we print every 3 steps
    log_destination.batch_end(dummy_state, logger)
    log_destination.training_end(dummy_state, logger)
    with open(log_file_name, 'r') as f:
        assert f.readlines() == [
            '[FIT][step=2]: { "metric": "fit", }\n',
            '[EPOCH][step=2]: { "metric": "epoch", }\n',
            '[BATCH][step=2]: { "metric": "batch", }\n',
            '[EPOCH][step=3]: { "metric": "epoch2", }\n',
        ]


class TestCoreLogger:

    @pytest.mark.parametrize("rank", [0, 1])
    def test_deferred(self, dummy_state_without_rank: State, log_file_name: str, monkeypatch: MonkeyPatch,
                      log_destination: FileLoggerBackend, rank: int):
        dummy_state = dummy_state_without_rank
        dummy_state.step = 2
        dummy_state.epoch = 0
        logger = Logger(dummy_state, backends=[log_destination])
        logger.metric_batch({"metric": "before_training_start"})
        monkeypatch.setattr(dist, "get_rank", lambda: rank)
        log_destination.training_start(dummy_state, logger)
        logger.metric_batch({"metric": "after_training_start"})
        log_destination.batch_end(dummy_state, logger)
        log_destination.training_end(dummy_state, logger)
        if rank == 0:
            with open(log_file_name, 'r') as f:
                assert f.readlines() == [
                    '[BATCH][step=2]: { "metric": "before_training_start", }\n',
                    '[BATCH][step=2]: { "metric": "after_training_start", }\n',
                ]
            return
        else:
            assert rank == 1
            assert not os.path.exists(log_file_name), "nothing should be logged on rank 1"

    def test_deep_copy(self, dummy_state_without_rank: State, log_destination: FileLoggerBackend,
                       monkeypatch: MonkeyPatch, log_file_name: str):
        # This test ensures that the logger deepcopies the logged metric when using deferred logging
        dummy_state = dummy_state_without_rank
        dummy_state.step = 2
        dummy_state.epoch = 0
        logger = Logger(dummy_state, backends=[log_destination])
        metric_data = [["hello"]]
        logger.metric_batch({"metric": metric_data})
        metric_data[0] = ["world"]
        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        log_destination.training_start(dummy_state, logger)
        logger.metric_batch({"metric": metric_data})
        log_destination.batch_end(dummy_state, logger)
        log_destination.training_end(dummy_state, logger)
        with open(log_file_name, 'r') as f:
            assert f.readlines() == [
                '[BATCH][step=2]: { "metric": [["hello"]], }\n',
                '[BATCH][step=2]: { "metric": [["world"]], }\n',
            ]
