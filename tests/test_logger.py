# Copyright 2021 MosaicML. All Rights Reserved.

import datetime
import os
import pathlib
from unittest.mock import MagicMock

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from tqdm import auto

from composer.core.event import Event
from composer.core.state import State
from composer.core.time import Time, Timestamp
from composer.loggers import (FileLoggerHparams, InMemoryLogger, Logger, LoggerDestination, LogLevel, TQDMLoggerHparams,
                              WandBLoggerHparams)
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import dist, reproducibility


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


@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
def test_tqdm_logger(composer_trainer_hparams: TrainerHparams, monkeypatch: MonkeyPatch, world_size: int):
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

    monkeypatch.setattr(auto, "tqdm", get_mock_tqdm)

    max_epochs = 2
    composer_trainer_hparams.max_duration = f"{max_epochs}ep"
    composer_trainer_hparams.loggers = [TQDMLoggerHparams()]
    trainer = composer_trainer_hparams.initialize_object()
    trainer.fit()
    if dist.get_global_rank() == 1:
        return
    assert len(is_train_to_mock_tqdms[True]) == max_epochs
    assert composer_trainer_hparams.validate_every_n_batches < 0
    assert len(is_train_to_mock_tqdms[False]) == composer_trainer_hparams.validate_every_n_epochs * max_epochs
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
def test_wandb_logger(composer_trainer_hparams: TrainerHparams, world_size: int):
    pytest.importorskip("wandb", reason="wandb is an optional dependency")
    del world_size  # unused. Set via launcher script
    composer_trainer_hparams.loggers = [
        WandBLoggerHparams(log_artifacts=True, log_artifacts_every_n_batches=1, extra_init_params={"mode": "disabled"})
    ]
    trainer = composer_trainer_hparams.initialize_object()
    trainer.fit()


def test_in_memory_logger(dummy_state: State):
    in_memory_logger = InMemoryLogger(LogLevel.EPOCH)
    logger = Logger(dummy_state, destinations=[in_memory_logger])
    logger.data_batch({"batch": "should_be_ignored"})
    logger.data_epoch({"epoch": "should_be_recorded"})
    dummy_state.timer.on_batch_complete(samples=1, tokens=1)
    logger.data_epoch({"epoch": "should_be_recorded_and_override"})

    # no batch events should be logged, since the level is epoch
    assert "batch" not in in_memory_logger.data
    assert len(in_memory_logger.data["epoch"]) == 2

    # `in_memory_logger.data` should contain everything
    timestamp, _, data = in_memory_logger.data["epoch"][0]
    assert timestamp.batch == 0
    assert data == "should_be_recorded"
    timestamp, _, data = in_memory_logger.data["epoch"][1]
    assert timestamp.batch == 1
    assert data == "should_be_recorded_and_override"

    # the most recent values should have just the last call to epoch
    assert in_memory_logger.most_recent_values["epoch"] == "should_be_recorded_and_override"
    assert in_memory_logger.most_recent_timestamps["epoch"].batch == 1


def test_in_memory_logger_get_timeseries():
    in_memory_logger = InMemoryLogger(LogLevel.BATCH)
    data = {"accuracy/val": [], "batch": [], "batch_in_epoch": []}
    for i in range(10):
        batch = i
        batch_in_epoch = i % 3
        timestamp = Timestamp(
            epoch=Time(0, "ep"),
            batch=Time(batch, "ba"),
            batch_in_epoch=Time(batch_in_epoch, "ba"),
            sample=Time(0, "sp"),
            sample_in_epoch=Time(0, "sp"),
            token=Time(0, "tok"),
            token_in_epoch=Time(0, "tok"),
        )
        state = MagicMock()
        state.timer.get_timestamp.return_value = timestamp
        datapoint = i / 3
        in_memory_logger.log_data(state=state, log_level=LogLevel.BATCH, data={"accuracy/val": datapoint})
        data["accuracy/val"].append(datapoint)
        data["batch"].append(batch)
        data["batch_in_epoch"].append(batch_in_epoch)

    timeseries = in_memory_logger.get_timeseries("accuracy/val")
    for k, v in data.items():
        assert np.all(timeseries[k] == np.array(v))


@pytest.mark.world_size(2)
def test_logger_run_name(dummy_state: State):
    # need to manually initialize distributed if not already initialized, since this test occurs outside of the trainer
    if not dist.is_initialized():
        dist.initialize_dist('gloo', timeout=datetime.timedelta(seconds=5))
    # seeding with the global rank to ensure that each rank has a different seed
    reproducibility.seed_all(dist.get_global_rank())

    logger = Logger(state=dummy_state)
    # The run name should be the same on every rank -- it is set via a distributed reduction
    # Manually verify that all ranks have the same run name
    run_names = dist.all_gather_object(logger.run_name)
    assert len(run_names) == 2  # 2 ranks
    assert all(run_name == run_names[0] for run_name in run_names)


def test_logger_file_artifact(dummy_state: State):

    file_logged = False

    class DummyLoggerDestination(LoggerDestination):

        def log_file_artifact(self, state: State, log_level: LogLevel, artifact_name: str, file_path: pathlib.Path, *,
                              overwrite: bool):
            nonlocal file_logged
            file_logged = True
            assert artifact_name == "foo"
            assert file_path.name == "bar"
            assert overwrite

    logger = Logger(state=dummy_state, destinations=[DummyLoggerDestination()])
    logger.file_artifact(
        log_level="epoch",
        artifact_name="foo",
        file_path="bar",
        overwrite=True,
    )

    assert file_logged
