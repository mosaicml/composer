# Copyright 2021 MosaicML. All Rights Reserved.

import datetime
import os

import pytest
import torch
import torch.backends.cudnn
import torch.distributed

from composer.utils import dist

DIST_TIMEOUT = datetime.timedelta(seconds=15)


@pytest.fixture(autouse=True)
def configure_dist(request: pytest.FixtureRequest):
    is_deepspeed = None
    is_gpu = None

    for item in request.session.items:
        item_is_gpu = item.get_closest_marker('gpu') is not None
        if is_gpu is None:
            is_gpu = item_is_gpu
        assert is_gpu == item_is_gpu
        item_is_deepspeed = item.get_closest_marker('deepspeed') is not None
        if is_deepspeed is None:
            is_deepspeed = item_is_deepspeed
        assert is_deepspeed == item_is_deepspeed

    if is_deepspeed and is_gpu:
        pytest.fail('Tests should be marked as deepspeed or gpu, not both. Deepspeed tests will run on a gpu.')

    dist_env_variable_names = ("NODE_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK")

    is_missing_all_dist_env_vars = all(x not in os.environ for x in dist_env_variable_names)

    if is_missing_all_dist_env_vars and (not is_deepspeed):
        # no distributed -- i.e. running pytest on the CLI
        # deepspeed still requires distributed
        return

    backend = "nccl" if (is_gpu or is_deepspeed) else "gloo"
    if not torch.distributed.is_initialized():
        dist.initialize_dist(backend, timeout=DIST_TIMEOUT)


@pytest.fixture(autouse=True)
def wait_for_all_procs(configure_dist: None):
    yield
    if not 'WORLD_SIZE' in os.environ:
        # Not running in a DDP environment
        return
    if int(os.environ['WORLD_SIZE']) == 1:
        # With just one proc, no need for the barrier
        return
    dist.barrier()
