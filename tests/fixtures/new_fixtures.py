# Copyright 2021 MosaicML. All Rights Reserved.

"""These fixtures are shared globally across the test suite."""

import datetime
import os

import pytest
import torch.distributed as dist
from torch.utils.data import DataLoader

from composer.cli.launcher import get_free_tcp_port
from composer.core import Logger, State
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


@pytest.fixture
def init_process_group(monkeypatch):
    """Use this fixture when initializing dist is needed outside of the Trainer's codepath."""

    if "RANK" not in os.environ:
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(get_free_tcp_port()))

    if not dist.is_initialized():
        dist.init_process_group("gloo", timeout=datetime.timedelta(10))

    yield

    if dist.is_initialized() and dist.is_available():
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")

    yield
