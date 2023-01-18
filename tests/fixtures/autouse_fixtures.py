# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import pathlib

import pytest
import tqdm.std

import composer
from composer.devices import DeviceCPU, DeviceGPU
from composer.utils import dist, reproducibility


@pytest.fixture(autouse=True)
def disable_tokenizer_parallelism():
    """This fixture prevents the below warning from appearing in tests:

        huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
        To disable this warning, you can either:
                - Avoid using `tokenizers` before the fork if possible
                - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


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


@pytest.fixture(autouse=True)
def chdir_to_tmp_path(tmp_path: pathlib.Path):
    os.chdir(tmp_path)


@pytest.fixture(autouse=True, scope='session')
def disable_tqdm_bars():
    # Disable tqdm progress bars globally in tests
    original_tqdm_init = tqdm.std.tqdm.__init__

    def new_tqdm_init(*args, **kwargs):
        if 'disable' not in kwargs:
            kwargs['disable'] = True
        return original_tqdm_init(*args, **kwargs)

    # Not using pytest monkeypatch as it is a function-scoped fixture
    tqdm.std.tqdm.__init__ = new_tqdm_init


@pytest.fixture(autouse=True)
def set_loglevels():
    """Ensures all log levels are set to DEBUG."""
    logging.basicConfig()
    logging.getLogger(composer.__name__).setLevel(logging.DEBUG)


@pytest.fixture(autouse=True)
def seed_all(rank_zero_seed: int, monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch reproducibility get_random_seed to always return the rank zero seed, and set the random seed before
    each test to the rank local seed."""
    monkeypatch.setattr(reproducibility, 'get_random_seed', lambda: rank_zero_seed)
    reproducibility.seed_all(rank_zero_seed + dist.get_global_rank())
