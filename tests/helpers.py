import multiprocessing as mp
import os
from functools import wraps
from typing import Callable, List

import pytest

DISTRIBUTED_INITIAL_TIMEOUT = 300
DISTRIBUTED_SUBSEQUENT_TIMEOUT = 5


def with_distributed(num_procs: List[int],
                     test_cpu: bool,
                     test_gpu: bool,
                     initial_timeout=30,
                     subsequent_timeout_per_device=5):

    if not test_cpu and not test_gpu:
        pytest.fail("Invalid test: must test on at least one of CPU and GPU!")

    cpu_params = []
    if test_cpu:
        cpu_params = [pytest.param(num_procs, False, id=f"{num_procs}-cpu") for num_procs in num_procs]

    gpu_params = []
    if test_gpu:
        gpu_params = [
            pytest.param(num_procs, True, marks=[pytest.mark.n_gpus(num_procs)], id=f"{num_procs}-gpu")
            for num_procs in num_procs
        ]

    def with_distributed_decorator(run_func: Callable):

        def dist_run_func_target(rank: int, num_procs: int, *args, **kwargs):
            print('within', rank, args)
            os.environ['WORLD_SIZE'] = str(num_procs)
            os.environ['RANK'] = str(rank)
            os.environ['LOCAL_RANK'] = str(rank)
            os.environ['LOCAL_WORLD_SIZE'] = str(num_procs)

            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = str(29400)

            run_func(num_procs=num_procs, *args, **kwargs)

        @pytest.mark.parametrize("num_procs,is_gpu", [*cpu_params, *gpu_params])
        @wraps(run_func)
        def dist_run_func(num_procs: int, is_gpu: bool, *args, **kwargs):

            processes = []
            ctx = mp.get_context('spawn')
            for rank in range(num_procs):
                _args = (rank, num_procs, is_gpu, *args)
                print('launching', rank, args)
                p = ctx.Process(target=dist_run_func_target, args=_args, kwargs=kwargs)
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

        return dist_run_func

    return with_distributed_decorator
