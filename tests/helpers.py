import multiprocessing as mp
import os
from typing import Callable, List

import pytest


def dist_run_target(target: Callable, rank: int, num_procs: int, *args, **kwargs):
    os.environ['WORLD_SIZE'] = str(num_procs)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(num_procs)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29400)

    target(*args, **kwargs)


def with_distributed(num_procs: int,
                     target: Callable,
                     initial_timeout: int = 30,
                     subsequent_timeout_per_device: int = 5):

    def run_target(*args, **kwargs):
        processes = []
        ctx = mp.get_context('spawn')
        for rank in range(num_procs):
            _args = (target, rank, num_procs, *args)
            p = ctx.Process(target=dist_run_target, args=_args, kwargs=kwargs)
            p.start()
            processes += [p]

        processes[0].join(initial_timeout)
        for p in processes[1:]:
            p.join(subsequent_timeout_per_device)

        for rank, p in enumerate(processes):
            if p.exitcode is None:
                p.kill()
                pytest.fail(f'Process {rank} did not exit', pytrace=False)
            elif p.exitcode != 0:
                pytest.fail(f'Process {rank} failed with error code {p.exitcode}', pytrace=False)

    return run_target
