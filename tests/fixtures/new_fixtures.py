# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""These fixtures are shared globally across the test suite."""
import datetime
import os
import shutil

import pytest
from torch.utils.data import DataLoader

from composer.core import State
from composer.loggers import Logger
from composer.trainer.devices import DeviceCPU, DeviceGPU
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.fixture
def minimal_state(rank_zero_seed: int):
    """Most minimally defined state possible.

    Tests should configure the state for their specific needs.
    """
    return State(
        model=SimpleModel(),
        run_name='minimal_run_name',
        rank_zero_seed=rank_zero_seed,
        max_duration='100ep',
        dataloader=DataLoader(RandomClassificationDataset()),
        dataloader_label='train',
    )


@pytest.fixture
def empty_logger(minimal_state: State) -> Logger:
    """Logger without any output configured."""
    return Logger(state=minimal_state, destinations=[])


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    monkeypatch.setenv('WANDB_START_METHOD', 'thread')
    if request.node.get_closest_marker('remote') is None:
        monkeypatch.setenv('WANDB_MODE', 'disabled')
    else:
        if not os.environ.get('WANDB_PROJECT'):
            monkeypatch.setenv('WANDB_PROJECT', 'pytest')


@pytest.fixture(autouse=True)
def configure_dist(request: pytest.FixtureRequest):
    # Configure dist globally when the world size is greater than 1,
    # so individual tests that do not use the trainer
    # do not need to worry about manually configuring dist.

    if dist.get_world_size() == 1:
        return

    device = DeviceCPU() if request.node.get_closest_marker('gpu') is None else DeviceGPU()
    if not dist.is_initialized():
        dist.initialize_dist(device, timeout=datetime.timedelta(seconds=300))
    # Hold PyTest until all ranks have reached this barrier. Ensure that no rank starts
    # any test before other ranks are ready to start it, which could be a cause of random timeouts
    # (e.g. rank 1 starts the next test while rank 0 is finishing up the previous test).
    # Fixtures are excluded from timeouts (see pyproject.toml)
    dist.barrier()


# Class-scoped temporary directory. That deletes itself. This is useful for e.g. not
# writing too many checkpoints.
@pytest.fixture(scope='class')
def self_destructing_tmp(tmp_path_factory: pytest.TempPathFactory):
    my_tmp_path = tmp_path_factory.mktemp('checkpoints')
    yield my_tmp_path
    shutil.rmtree(str(my_tmp_path))
