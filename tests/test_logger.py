# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib
from unittest.mock import MagicMock

import pytest
import torch.distributed as dist
import tqdm
from _pytest.monkeypatch import MonkeyPatch

from composer.core.event import Event
from composer.core.logging import Logger, LogLevel
from composer.core.state import State
from composer.loggers.file_logger import FileLoggerBackend
from composer.loggers.logger_hparams import (FileLoggerBackendHparams, TQDMLoggerBackendHparams,
                                             WandBLoggerBackendHparams)
from composer.trainer.trainer_hparams import TrainerHparams


@pytest.fixture
def log_file_name(tmpdir: pathlib.Path) -> str:
    return os.path.join(tmpdir, "output.log")


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
    dummy_state.timer.on_batch_complete()
    dummy_state.timer.on_batch_complete()
    dummy_state.timer.on_epoch_complete()
    dummy_state.timer.on_epoch_complete()
    logger = Logger(dummy_state, backends=[log_destination])
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    log_destination.run_event(Event.INIT, dummy_state, logger)
    logger.metric_fit({"metric": "fit"})  # should print
    logger.metric_epoch({"metric": "epoch"})  # should print
    logger.metric_batch({"metric": "batch"})  # should print
    logger.metric_verbose({"metric": "verbose"})  # should NOT print, since we're on the BATCH log level
    dummy_state.timer.on_epoch_complete()
    logger.metric_epoch({"metric": "epoch1"})  # should NOT print, since we print every 2 epochs
    dummy_state.timer.on_epoch_complete()
    dummy_state.timer.on_batch_complete()
    log_destination.run_event(Event.BATCH_END, dummy_state, logger)
    logger.metric_epoch({"metric": "epoch2"})  # should print
    logger.metric_batch({"metric": "batch1"})  # should NOT print, since we print every 3 steps
    log_destination.run_event(Event.BATCH_END, dummy_state, logger)
    log_destination.run_event(Event.TRAINING_END, dummy_state, logger)
    with open(log_file_name, 'r') as f:
        assert f.readlines() == [
            '[FIT][step=2]: { "metric": "fit", }\n',
            '[EPOCH][step=2]: { "metric": "epoch", }\n',
            '[BATCH][step=2]: { "metric": "batch", }\n',
            '[EPOCH][step=3]: { "metric": "epoch2", }\n',
        ]


def test_tqdm_logger(mosaic_trainer_hparams: TrainerHparams, monkeypatch: MonkeyPatch):
    is_train_to_mock_tqdms = {
        True: [],
        False: [],
    }

    def get_mock_tqdm(position: int, *args, **kwargs):
        del args, kwargs  # unused
        is_train = position == 0
        mock_tqdm = MagicMock()
        is_train_to_mock_tqdms[is_train].append(mock_tqdm)
        return mock_tqdm

    monkeypatch.setattr(tqdm, "tqdm", get_mock_tqdm)
    max_epochs = 2
    mosaic_trainer_hparams.max_duration = f"{max_epochs}ep"
    mosaic_trainer_hparams.loggers = [TQDMLoggerBackendHparams()]
    trainer = mosaic_trainer_hparams.initialize_object()
    trainer.fit()
    assert len(is_train_to_mock_tqdms[True]) == max_epochs
    assert mosaic_trainer_hparams.validate_every_n_batches < 0
    assert len(is_train_to_mock_tqdms[False]) == mosaic_trainer_hparams.validate_every_n_epochs * max_epochs
    for mock_tqdm in is_train_to_mock_tqdms[True]:
        assert mock_tqdm.update.call_count == trainer.state.steps_per_epoch
        mock_tqdm.close.assert_called_once()
    for mock_tqdm in is_train_to_mock_tqdms[False]:
        assert mock_tqdm.update.call_count == trainer._eval_subset_num_batches
        mock_tqdm.close.assert_called_once()


@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.timeout(10)
def test_wandb_logger(mosaic_trainer_hparams: TrainerHparams, world_size: int):
    try:
        import wandb
        del wandb
    except ImportError:
        pytest.skip("wandb is not installed")
    del world_size  # unused. Set via launcher script
    mosaic_trainer_hparams.loggers = [
        WandBLoggerBackendHparams(log_artifacts=True,
                                  log_artifacts_every_n_batches=1,
                                  extra_init_params={"mode": "disabled"})
    ]
    trainer = mosaic_trainer_hparams.initialize_object()
    trainer.fit()
