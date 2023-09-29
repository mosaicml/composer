# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
import subprocess
import pathlib

import mcli
import pytest
import torch
import tqdm.std

import composer
from composer.devices import DeviceCPU, DeviceGPU
from composer.utils import dist, reproducibility
import psutil

@pytest.fixture(autouse=True)
def disable_tokenizer_parallelism():
    """This fixture prevents the below warning from appearing in tests:

        huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
        To disable this warning, you can either:
                - Avoid using `tokenizers` before the fork if possible
                - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@pytest.fixture(scope='function', autouse=True)
def disk_space_recorder(request):
    """This fixture records the disk space of each individual pytest function to help us debug when tests run out of disk space."""
    # Get the disk space before the test
    disk_space_before = get_disk_space()
    yield
    # Get the disk space after the test
    disk_space_after = get_disk_space()
    # Record the disk space to a file
    record_disk_space(disk_space_before, disk_space_after, request.node.name)

def get_disk_space():
    # Using psutil to get disk space
    return psutil.disk_usage('/')
    
def record_disk_space(before, after, test_name):
    change_in_free_space_bytes = before.free - after.free
    change_in_free_space_megabytes = change_in_free_space_bytes // (1024 * 1024)
    with open('/tmp/disk_space_report.txt', 'a') as file:
        file.write(f'Test: {test_name}\n')
        file.write(f'Disk space before: total: {before.total} bytes, ussed: {before.used} bytes, free: {before.free} bytes\n')
        file.write(f'Disk space after: total: {after.total} bytes, used: {after.used} bytes, free: {after.free} bytes\n')
        file.write(f'Change in free space: {change_in_free_space_bytes} bytes ({change_in_free_space_megabytes:.2f} MB)\n')
        if abs(change_in_free_space_megabytes) >= 1:
            file.write(f'Notice: Test {test_name} free disk space changed by 1 MB or more\n')

@pytest.fixture(autouse=True)
def clear_cuda_cache(request):
    """Clear memory between GPU tests."""
    marker = request.node.get_closest_marker('gpu')
    if marker is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()  # Only gc on GPU tests as it 2x slows down CPU tests


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


@pytest.fixture(autouse=True)
def mapi_fixture(monkeypatch):
    # Composer auto-adds mosaicml logger when running on platform. Disable logging for tests.
    mock_update = lambda *args, **kwargs: None
    monkeypatch.setattr(mcli, 'update_run_metadata', mock_update)


@pytest.fixture(autouse=True)
def remove_run_name_env_var():
    # Remove environment variables for run names in unit tests
    composer_run_name = os.environ.get('COMPOSER_RUN_NAME')
    run_name = os.environ.get('RUN_NAME')

    if 'COMPOSER_RUN_NAME' in os.environ:
        del os.environ['COMPOSER_RUN_NAME']
    if 'RUN_NAME' in os.environ:
        del os.environ['RUN_NAME']

    yield

    if composer_run_name is not None:
        os.environ['COMPOSER_RUN_NAME'] = composer_run_name
    if run_name is not None:
        os.environ['RUN_NAME'] = run_name
