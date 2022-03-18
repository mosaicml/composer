# Copyright 2021 MosaicML. All Rights Reserved.

"""These fixtures are shared globally across the test suite."""
import pytest
from torch.utils.data import DataLoader

from composer.core import State
from composer.loggers import Logger
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.fixture
def minimal_state(rank_zero_seed: int):
    """Most minimally defined state possible.

    Tests should configure the state for their specific needs.
    """
    return State(
        model=SimpleModel(),
        rank_zero_seed=rank_zero_seed,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        evaluators=[],
        max_duration='100ep',
    )


@pytest.fixture
def empty_logger(minimal_state: State) -> Logger:
    """Logger without any output configured."""
    return Logger(state=minimal_state, destinations=[])


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")

    yield
