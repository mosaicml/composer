# Copyright 2021 MosaicML. All Rights Reserved.

"""These fixtures are shared globally across the test suite."""
from typing import Callable

import pytest
from torch.utils.data import DataLoader

from composer.core import Logger, State
from composer.core.logging import LogLevel
from composer.core.time import Time, Timestamp
from composer.loggers import InMemoryLogger
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.fixture
def minimal_state():
    """Most minimally defined state possible.

    Tests should configure the state for their specific needs.
    """
    return State(
        model=SimpleModel(),
        train_dataloader=DataLoader(RandomClassificationDataset()),
        evaluators=[],
        max_duration='100ep',
    )


@pytest.fixture
def empty_logger(minimal_state: State) -> Logger:
    """Logger without any output configured."""
    return Logger(state=minimal_state, backends=[])


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")

    yield


@pytest.fixture
def in_memory_logger_populated(datafxn: Callable = lambda x: x / 3,
                               datafield: str = "accuracy/val",
                               n_epochs: int = 5,
                               batches_per_epoch: int = 6,
                               samples_per_batch: int = 10,
                               tokens_per_sample: int = 20,
                               loglevel: LogLevel = LogLevel.BATCH):
    logger = InMemoryLogger(loglevel)
    for batch in range(n_epochs * batches_per_epoch):
        datapoint = datafxn(batch)
        sample_in_epoch = (batch % batches_per_epoch) * samples_per_batch
        token = tokens_per_sample * samples_per_batch * batch
        token_in_epoch = (batch % batches_per_epoch) * samples_per_batch * tokens_per_sample
        timestamp = Timestamp(epoch=Time(batch // batches_per_epoch, "ep"),
                              batch=Time(batch, "ba"),
                              batch_in_epoch=Time(batch % batches_per_epoch, "ba"),
                              sample=Time(batch * samples_per_batch, "sp"),
                              sample_in_epoch=Time(sample_in_epoch, "sp"),
                              token=Time(token, "tok"),
                              token_in_epoch=Time(token_in_epoch, "tok"))
        logger.log_metric(timestamp=timestamp, log_level=loglevel, data={datafield: datapoint})
        return logger
