# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import datetime
import os
import warnings
from typing import Any, List, Optional, Sequence, TypeVar, cast

import torch
import torch.distributed as dist
import torch.utils.data

TObj = TypeVar("TObj")


def _get_distributed_config_var(env_var: str,
                                human_name: str,
                                default: int,
                                fetch_fn_name: Optional[str] = None) -> int:
    if not dist.is_available():
        warnings.warn("DistributedDefaultValueWarning: Torch distributed is not available; "
                      f"returning {default} for {human_name}")
        return default

    if dist.is_initialized() and fetch_fn_name is not None:
        dist_value = int(getattr(dist, fetch_fn_name)())
        if env_var in os.environ:
            env_value = int(os.environ[env_var])
            if dist_value != env_value:
                raise RuntimeError("Torch distributed has been initialized with a value of "
                                   f"{dist_value} for {human_name}, but environment variable "
                                   f"{env_var} has value {env_value}.")
        return dist_value

    if env_var in os.environ:
        return int(os.environ[env_var])

    if dist.is_initialized():
        raise RuntimeError("Torch distributed is initialized but environment variable "
                           f"{env_var} is not set.")

    warnings.warn(f"DistributedDefaultValueWarning: {env_var} env var not set and Torch "
                  f"distributed not initialized; returning {default} for {human_name}.")
    return default


def get_world_size() -> int:
    """Returns the world size, which is the number of processes participating in this training run.

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
    return _get_distributed_config_var(env_var="LOCAL_RANK", human_name="local rank", default=0)


def get_node_rank() -> int:
    """Returns the node rank. For example, if there are 2 nodes, and 2 ranks per node, then
    global ranks 0-1 will have a node rank of 0, and global ranks 2-3 will have a node rank of 1.

    Returns:
        int: The node rank, starting at 0.
    """
    return _get_distributed_config_var(env_var="NODE_RANK", human_name="node rank", default=0)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(f"The world_size({world_size}) > 1, but the distributed package is not "
                       "available or has not been initialized. Please check you have initialized "
                       "the distributed runtime and that PyTorch has been built with distributed "
                       "support.")


def all_reduce(
    tensor: torch.Tensor,
    reduce_operation: str = "SUM",
) -> None:
    if dist.is_available() and dist.is_initialized():
        reduce_op = getattr(dist.ReduceOp, reduce_operation.upper())
        dist.all_reduce(tensor, op=reduce_op)
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(f"The world_size({world_size}) > 1, but the distributed package is not "
                       "available or has not been initialized. Please check you have initialized "
                       "the distributed runtime and that PyTorch has been built with distributed "
                       "support.")


def broadcast(tensor: torch.Tensor, src: int) -> None:
    """Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes participating in the collective.
    See :meth:`torch.distributed.broadcast`.

    Args:
        tensor (torch.Tensor): Data to be sent if ``src`` is the rank of current process,
            and tensor to be used to save received data otherwise.
        src (int): Source rank
    """
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src)
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(f"The world_size({world_size}) > 1, but the distributed package is not "
                       "available or has not been initialized. Please check you have initialized "
                       "the distributed runtime and that PyTorch has been built with distributed "
                       "support.")


def broadcast_object_list(object_list: List[Any], src: int = 0) -> None:
    """Broadcasts picklable objects in ``object_list`` to the whole group.
    Similar to :meth:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be broadcasted.
    See :meth:`torch.distributed.broadcast`.

    Args:
        object_list (torch.Tensor): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will be broadcast,
            but each rank must provide lists of equal sizes.
        src (int, optional): Source rank (default: ``0``)
    """
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(object_list, src)
        # torch.distributed will replace the None's in obj_gather_list with the gathered objects on rank 0
        # or will just be None on non-rank-0
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(f"The world_size({world_size}) > 1, but the distributed package is not "
                       "available or has not been initialized. Please check you have initialized "
                       "the distributed runtime and that PyTorch has been built with distributed "
                       "support.")


def all_gather(tensor: torch.Tensor) -> Sequence[torch.Tensor]:
    """all_gather collects a tensor from each rank, and returns a sequence of tensors indexed by rank

    Args:
        tensor (torch.Tensor): tensor from each rank to be gathered

    Returns:
        Sequence[Tensor]: A sequence of tensors indexed by rank
    """
    if dist.is_available() and dist.is_initialized():
        obj_gather_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        dist.all_gather(obj_gather_list, tensor)
        return obj_gather_list
    world_size = get_world_size()
    if world_size == 1:
        return [tensor]
    raise RuntimeError(f"The world_size({world_size}) > 1, but the distributed package is not "
                       "available or has not been initialized. Please check you have initialized "
                       "the distributed runtime and that PyTorch has been built with distributed "
                       "support.")


def all_gather_object(obj: TObj) -> List[TObj]:
    """all_gather_object collects a pickleable object from each rank, and returns a list of
    these objects indexed by rank

    Args:
        obj (TObj): Object to be gathered

    Returns:
        List[TObj]: A list of objects indexed by rank
    """
    if dist.is_available() and dist.is_initialized():
        obj_gather_list = [None for _ in range(get_world_size())]
        dist.all_gather_object(obj_gather_list, obj)
        # torch.distributed will replace the None's in obj_gather_list with the gathered objects on rank 0
        # or will just be None on non-rank-0
        return cast(List[TObj], obj_gather_list)
    world_size = get_world_size()
    if world_size == 1:
        return [obj]
    raise RuntimeError(f"The world_size({world_size}) > 1, but the distributed package is not "
                       "available or has not been initialized. Please check you have initialized "
                       "the distributed runtime and that PyTorch has been built with distributed "
                       "support.")


def is_available():
    return dist.is_available()


def is_initialized():
    return dist.is_initialized()


def initialize_dist(backend: str, timeout: datetime.timedelta):
    if not dist.is_available():
        if get_world_size() != 1:
            raise RuntimeError("When the world size is > 1, ``torch.distributed`` must be used. However, it is "
                               "not available in your installation of PyTorch. Please install or build PyTorch "
                               "with distributed support.")
        return

    if dist.is_initialized():
        if dist.get_backend() != backend.lower():
            raise RuntimeError(f"The requested backend ({backend}) differs from the backend "
                               f"of the current process group ({dist.get_backend()}). If you "
                               "wish to change backends, please restart the python process.")
        return

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        warnings.warn("NoDistributedWarning: RANK and WORLD_SIZE env vars not set; assuming no "
                      "parallelization. If this is unexpected, make sure you are running your "
                      "training script with the composer CLI tool.")
    elif get_world_size() > 1:
        dist.init_process_group(backend, timeout=timeout)


def get_sampler(dataset, *, drop_last: bool, shuffle: bool) -> torch.utils.data.Sampler:
    return torch.utils.data.DistributedSampler[int](
        dataset,
        drop_last=drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
    )
