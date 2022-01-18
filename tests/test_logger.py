# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib
from unittest.mock import MagicMock

import pytest
import tqdm
from _pytest.monkeypatch import MonkeyPatch

from composer.core.event import Event
from composer.core.logging import Logger, LogLevel
from composer.core.state import State
from composer.loggers.logger_hparams import (FileLoggerBackendHparams, TQDMLoggerBackendHparams,
                                             WandBLoggerBackendHparams)
from composer.trainer.trainer_hparams import TrainerHparams


@pytest.fixture
def log_file_name(tmpdir: pathlib.Path) -> str:
    return os.path.join(tmpdir, "output.log")


@pytest.mark.parametrize("log_level", [LogLevel.EPOCH, LogLevel.BATCH])
def test_file_logger(dummy_state: State, log_level: LogLevel, log_file_name: str):
    log_destination = FileLoggerBackendHparams(
        log_interval=3,
        log_level=log_level,
        filename=log_file_name,
        buffer_size=1,
        flush_interval=1,
    ).initialize_object()
    dummy_state.timer.on_batch_complete()
    dummy_state.timer.on_batch_complete()
    dummy_state.timer.on_epoch_complete()
    logger = Logger(dummy_state, backends=[log_destination])
    log_destination.run_event(Event.INIT, dummy_state, logger)
    logger.metric_fit({"metric": "fit"})  # should print
    logger.metric_epoch({"metric": "epoch"})  # should print on batch level, since epoch calls are always printed
    logger.metric_batch({"metric": "batch"})  # should print on batch level, since we print every 3 steps
    dummy_state.timer.on_epoch_complete()
    logger.metric_epoch({"metric": "epoch1"})  # should print, since we log every 3 epochs
    dummy_state.timer.on_epoch_complete()
    dummy_state.timer.on_batch_complete()
    log_destination.run_event(Event.BATCH_END, dummy_state, logger)
    logger.metric_epoch({"metric": "epoch2"})  # should print on batch level, since epoch calls are always printed
    logger.metric_batch({"metric": "batch1"})  # should NOT print
    log_destination.run_event(Event.BATCH_END, dummy_state, logger)
    log_destination.run_event(Event.TRAINING_END, dummy_state, logger)
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


@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
def test_tqdm_logger(mosaic_trainer_hparams: TrainerHparams, monkeypatch: MonkeyPatch, world_size: int):
    del world_size  # unused. Set via launcher script
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
