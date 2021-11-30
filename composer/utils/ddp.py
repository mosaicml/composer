# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import collections.abc
import datetime
import os
import warnings
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Callable, ContextManager, List, Optional, Sequence, TypeVar, Union, cast

import torch
import torch.distributed as dist
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel

from composer.utils.iter_helpers import ensure_tuple
from composer.utils.string_enum import StringEnum

if TYPE_CHECKING:
    from composer.core.state import State
    from composer.core.types import DataLoader, Model
    from composer.datasets.dataloader import DataloaderHparams, DataloaderSpec

TObj = TypeVar("TObj")


class DDPSyncStrategy(StringEnum):
    """How and when DDP gradient synchronization should happen.

    Attributes:
        SINGLE_AUTO_SYNC: The default behavior for DDP. Gradients are synchronized as they
            computed, for only the final microbatch of a batch. This is the most efficient
            strategy, but can lead to errors when ``find_unused_parameters`` is set, since
            it is possible different microbatches may use different sets of parameters,
            leading to an incomplete sync.
        MULTI_AUTO_SYNC: The default behavior for DDP when ``find_unused_parameters`` is set.
            Gradients are synchronized as they are computed for all microbatches. This ensures
            complete synchronization, but is less efficient than :attr:`SINGLE_AUTO_SYNC`. This
            efficiency gap is usually small, as long as either DDP syncs are a small portion
            of the trainer's overall runtime, or the number of microbatches per batch is
            relatively small.
        FORCED_SYNC: Gradients are manually synchronized only after all gradients have been
            computed for the final microbatch of a batch. Like :attr:`MULTI_AUTO_SYNC`, this
            strategy ensures complete gradient synchronization, but this tends to be slower than
            :attr:`MULTI_AUTO_SYNC`. This is because ordinarily syncs can happen in parallel
            with the ``loss.backward()`` computation, meaning syncs can be mostly complete by
            the time that function finishes. However, in certain circumstances, syncs may take
            a very long time to complete - if there are also a lot of microbatches per batch,
            this strategy may be optimal.
    """
    SINGLE_AUTO_SYNC = "single_auto_sync"
    MULTI_AUTO_SYNC = "multi_auto_sync"
    FORCED_SYNC = "forced_sync"


def _get_distributed_config_var(env_var: str,
                                human_name: str,
                                default: int,
                                fetch_fn_name: Optional[str] = None) -> int:
    if not dist.is_available():
        warnings.warn(
            f"DDPDefaultValueWarning: Torch distributed is not available; returning {default} for {human_name}")
        return default

    if not env_var in os.environ:
        warnings.warn(f"DDPDefaultValueWarning: {env_var} env var not set"
                      f"{' and process group not initialized' if fetch_fn_name is not None else ''}; "
                      f"returning {default} for {human_name}.")
        env_value = default
    else:
        env_value = int(os.environ[env_var])

    if dist.is_initialized() and fetch_fn_name is not None:
        assert env_value == int(getattr(dist, fetch_fn_name)()), "invariant violation"

    return env_value


def get_world_size() -> int:
    """Returns the DDP world size

    Returns:
        int: The world size
    """
    return _get_distributed_config_var(env_var="WORLD_SIZE",
                                       human_name="world size",
                                       default=1,
                                       fetch_fn_name="get_world_size")


def get_global_rank() -> int:
    """Returns the global rank of the current process, which is in `[0, WORLD_SIZE - 1]`

    Returns:
        int: The global rank
    """
    return _get_distributed_config_var(env_var="RANK", human_name="global rank", default=0, fetch_fn_name="get_rank")


def get_local_world_size() -> int:
    """Returns the local world size, which is the number of processes for the current node.

    Returns:
        int: The local world size
    """
    return _get_distributed_config_var(env_var="LOCAL_WORLD_SIZE", human_name="local world size", default=1)


def get_local_rank() -> int:
    """Returns the local rank for the current process, which is in `[0, LOCAL_WORLD_SIZE - 1]`

    Returns:
        int: The local world size
    """
    local_rank = _get_distributed_config_var(env_var="LOCAL_RANK", human_name="local rank", default=0)
    assert local_rank == get_global_rank() % get_local_world_size(), "invariant violation"
    return local_rank


def barrier() -> None:
    if dist.is_available():
        dist.barrier()
    # If not on DDP, then do nothing


def all_reduce(
    tensor: torch.Tensor,
    reduce_operation: str = "SUM",
) -> None:
    if dist.is_available():
        reduce_op = getattr(dist.ReduceOp, reduce_operation.upper())
        dist.all_reduce(tensor, op=reduce_op)
    else:
        raise NotImplementedError("Non-DDP versions of reduce operations are not yet implemented")


def all_gather(tensor: torch.Tensor) -> Sequence[torch.Tensor]:
    """gather_to_rank_zero collects a tensor from each rank, and returns a sequence of tensors indexed by rank

    Args:
        tensor (torch.Tensor): tensor from each rank to be gathered

    Returns:
        Sequence[Tensor]: A sequence of tensors indexed by rank
    """
    if dist.is_available():
        obj_gather_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        dist.all_gather(obj_gather_list, tensor)
        return obj_gather_list
    else:
        return [tensor]


def all_gather_object(obj: TObj) -> List[TObj]:
    """gather_object_to_rank_zero collects a pickleable object from each rank, and returns a list of
    these objects indexed by rank

    Args:
        obj (TObj): Object to be gathered

    Returns:
        List[TObj]: A list of objects indexed by rank
    """
    if dist.is_available():
        obj_gather_list = [None for _ in range(get_world_size())]
        dist.all_gather_object(obj_gather_list, obj)
        # torch.distributed will replace the None's in obj_gather_list with the gathered objects on rank 0
        # or will just be None on non-rank-0
        return cast(List[TObj], obj_gather_list)
    else:
        return [obj]


def initialize_ddp(backend: str, timeout: datetime.timedelta):
    if not dist.is_available():
        return
    if dist.is_initialized():

        if not dist.get_backend() == backend.lower():
            raise RuntimeError(
                f"The requested backend ({backend}) differs from the backend "
                "of the current process group ({torch.distributed.get_backend()}). If you wish to change backends, "
                "please restart the python process.")
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Assume we can initialize based off of env vars
        dist.init_process_group(backend, timeout=timeout)
        return

    warnings.warn("NoDDPWarning: RANK and WORLD_SIZE env vars not set; assuming no parallelization. "
                  "If this is unexpected, make sure you are running your training script with the "
                  "composer executable.")
    store = dist.HashStore()

    dist.init_process_group(backend, timeout=timeout, store=store, world_size=1, rank=0)


def prepare_module(module: Model, find_unused_parameters: bool) -> Model:
    if dist.is_available():
        if any((p.requires_grad for p in module.parameters())):
            ddp_model = DistributedDataParallel(module, find_unused_parameters=find_unused_parameters)
            return ddp_model
        return module
    else:
        return module


def create_dataloader(batch_size: int, dataloader_hparams: DataloaderHparams,
                      dataloader_spec: DataloaderSpec) -> DataLoader:
    # TODO(ravi) refactor this function to return a sampler rather than create the dataloader
    from composer.datasets.dataloader import DDPDataLoader
    if dist.is_available():
        sampler = torch.utils.data.DistributedSampler[int](dataloader_spec.dataset,
                                                           drop_last=dataloader_spec.drop_last,
                                                           shuffle=dataloader_spec.shuffle)
    else:
        assert isinstance(dataloader_spec.dataset, collections.abc.Sized)
        sampler = torch.utils.data.RandomSampler(dataloader_spec.dataset, generator=dataloader_spec.generator)
    dataloader = dataloader_hparams.initialize_object(batch_size, sampler, dataloader_spec)
    if dist.is_available():
        dataloader = DDPDataLoader(dataloader)
    return dataloader


@contextmanager
def sync_context(state: State, is_final_microbatch: bool, sync_strategy: Union[str, DDPSyncStrategy]):
    assert isinstance(state.model, DistributedDataParallel), "state.model is not wrapped by DDP"
    assert state.optimizers is not None, "optimizers have not been initialized"
    sync_strategy = DDPSyncStrategy(sync_strategy)

    no_sync_context = cast(Callable[[], ContextManager], state.model.no_sync)
    auto_sync_context = nullcontext

    if sync_strategy == DDPSyncStrategy.SINGLE_AUTO_SYNC:
        context = auto_sync_context if is_final_microbatch else no_sync_context
        with context():
            yield

    elif sync_strategy == DDPSyncStrategy.MULTI_AUTO_SYNC:
        with auto_sync_context():
            yield

    elif sync_strategy == DDPSyncStrategy.FORCED_SYNC:
        try:
            with no_sync_context():
                yield
        finally:
            if is_final_microbatch:
                for optimizer in ensure_tuple(state.optimizers):
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            if p.grad is not None:
                                all_reduce(p.grad)
                                p.grad = p.grad / get_world_size()

    else:
        raise ValueError("Unknown sync strategy", sync_strategy)
