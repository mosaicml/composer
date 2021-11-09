# Copyright 2021 MosaicML. All Rights Reserved.

import multiprocessing as mp
import os
import time
from typing import Callable

import pytest
import torch.distributed


@pytest.fixture(autouse=True)
def ddp_cleanup():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _dist_run_target(target: Callable, rank: int, num_procs: int, *args, **kwargs):
    os.environ['WORLD_SIZE'] = str(num_procs)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(num_procs)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29400)

    target(*args, **kwargs)


def with_distributed(num_procs: int, target: Callable, timeout: int = 30):

    def run_target(*args, **kwargs):
        processes = []
        ctx = mp.get_context('spawn')
        for rank in range(num_procs):
            _args = (target, rank, num_procs, *args)
            p = ctx.Process(target=_dist_run_target, args=_args, kwargs=kwargs)
            p.start()
            processes.append(p)

        start_time = time.time()

        while any([p.is_alive() for p in processes]):
            has_timed_out = time.time() - start_time > timeout
            for rank, p in enumerate(processes):
                if p.is_alive() and not has_timed_out:
                    continue
                if p.exitcode is None:
                    p.kill()
                    pytest.fail(f'Process {rank} did not exit', pytrace=False)
                elif p.exitcode != 0:
                    pytest.fail(f'Process {rank} failed with error code {p.exitcode}', pytrace=False)
            time.sleep(0.5)

    return run_target
