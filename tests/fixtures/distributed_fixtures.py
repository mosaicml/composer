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

    if is_deepspeed:
        assert not is_gpu
        if not "RANK" in os.environ:
            os.environ["RANK"] = str(0)
            os.environ["LOCAL_RANK"] = str(0)
            os.environ["WORLD_SIZE"] = str(1)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(26000)
        import deepspeed
        deepspeed.init_distributed(timeout=DIST_TIMEOUT)
        return

    assert not is_deepspeed
    backend = "nccl" if is_gpu else "gloo"
    if not torch.distributed.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            torch.distributed.init_process_group(backend, timeout=DIST_TIMEOUT)
        else:
            store = torch.distributed.HashStore()
            torch.distributed.init_process_group(backend, timeout=DIST_TIMEOUT, store=store, world_size=1, rank=0)


@pytest.fixture(autouse=True)
def wait_for_all_procs(subfolder_run_directory: None, configure_dist: None):
    yield
    if not 'WORLD_SIZE' in os.environ:
        # Not running in a DDP environment
        return
    if int(os.environ['WORLD_SIZE']) == 1:
        # With just one proc, no need for the barrier
        return
    dist.barrier()
