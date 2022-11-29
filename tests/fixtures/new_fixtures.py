# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""These fixtures are shared globally across the test suite."""
import os
import time

import coolname
import pytest
import torch
from torch.utils.data import DataLoader

from composer.core import State
from composer.devices import DeviceCPU, DeviceGPU
from composer.loggers import Logger
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
        monkeypatch.setenv('WANDB_MODE', 'offline')
    else:
        if not os.environ.get('WANDB_PROJECT'):
            monkeypatch.setenv('WANDB_PROJECT', 'pytest')


@pytest.fixture(autouse=True, scope='session')
def configure_dist(request: pytest.FixtureRequest):
    # Configure dist globally when the world size is greater than 1,
    # so individual tests that do not use the trainer
    # do not need to worry about manually configuring dist.

    if dist.get_world_size() == 1:
        return

    device = None

    for item in request.session.items:
        device = DeviceCPU() if item.get_closest_marker('gpu') is None else DeviceGPU()
        break

    assert device is not None

    if not dist.is_initialized():
        dist.initialize_dist(device, timeout=300.0)
    # Hold PyTest until all ranks have reached this barrier. Ensure that no rank starts
    # any test before other ranks are ready to start it, which could be a cause of random timeouts
    # (e.g. rank 1 starts the next test while rank 0 is finishing up the previous test).
    dist.barrier()


@pytest.fixture(scope='session')
def test_session_name(configure_dist: None) -> str:
    """Generate a random name for the test session that is the same on all ranks."""
    del configure_dist  # unused
    generated_session_name = str(int(time.time())) + '-' + coolname.generate_slug(2)
    name_list = [generated_session_name]
    # ensure all ranks have the same name
    dist.broadcast_object_list(name_list)
    return name_list[0]


@pytest.fixture()
def dummy_state(
    rank_zero_seed: int,
    request: pytest.FixtureRequest,
) -> State:

    model = SimpleModel()
    if request.node.get_closest_marker('gpu') is not None:
        # If using `dummy_state`, then not using the trainer, so move the model to the correct device
        model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    state = State(
        model=model,
        run_name='dummy_run_name',
        precision='fp32',
        grad_accum=1,
        rank_zero_seed=rank_zero_seed,
        optimizers=optimizer,
        max_duration='10ep',
    )
    state.schedulers = scheduler
    state.set_dataloader(DataLoader(RandomClassificationDataset()), 'train')

    return state
