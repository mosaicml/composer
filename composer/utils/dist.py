# Copyright 2021 MosaicML. All Rights Reserved.

"""Helper methods for :mod:`torch.distributed`.

To use :mod:`torch.distributed`, launch your training script with the
:ref:`composer launcher for distributed training <distributed-training>`. For example,
the following command launches an eight-process training run.

.. code-block::

    composer -n 8 path/to/train.py

The composer launcher will automatically configure the following environment variables, which are
required for distributed training:

* ``RANK``: The global rank of the process, which should be on ``[0; WORLD_SIZE - 1]``.
* ``LOCAL_RANK``: The local rank for the process, which should be on ``[0; LOCAL_WORLD_SIZE - 1]``.
* ``NODE_RANK``: The rank of the node.
* ``WORLD_SIZE``: The total number of processes.
* ``LOCAL_WORLD_SIZE``: The number of processes on the current node.
* ``MASTER_ADDR``: The hostname for the rank-zero process.
* ``MASTER_PORT``: The port for the rank-zero process.

If none of these environment variables are set, this module will safely assume a single-rank configuration, where::

    RANK=0
    LOCAL_RANK=0
    NODE_RANK=0
    WORLD_SIZE=1
    LOCAL_WORLD_SIZE=1
"""
from __future__ import annotations

import datetime
import os
import textwrap
import warnings
from typing import Any, List, Optional, Sequence, TypeVar, cast

import torch
import torch.distributed as dist
import torch.utils.data

TObj = TypeVar("TObj")

__all__ = [
    "all_gather",
    "all_gather_object",
    "all_reduce",
    "barrier",
    "broadcast",
    "broadcast_object_list",
    "get_global_rank",
    "get_local_rank",
    "get_local_world_size",
    "get_node_rank",
    "get_sampler",
    "get_world_size",
    "initialize_dist",
    "is_available",
    "is_initialized",
]


def _get_distributed_config_var(
    env_var: str,
    human_name: str,
    default: int,
    fetch_fn_name: Optional[str] = None,
) -> int:
    if not dist.is_available():
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

    return default


def get_world_size() -> int:
    """Returns the world size, which is the number of processes participating in this training run.

    Returns:
        int: The world size.
    """
    return _get_distributed_config_var(env_var="WORLD_SIZE",
                                       human_name="world size",
                                       default=1,
                                       fetch_fn_name="get_world_size")


def get_global_rank() -> int:
    """Returns the global rank of the current process, which is on ``[0; WORLD_SIZE - 1]``.

    Returns:
        int: The global rank.
    """
    return _get_distributed_config_var(env_var="RANK", human_name="global rank", default=0, fetch_fn_name="get_rank")


def get_local_world_size() -> int:
    """Returns the local world size, which is the number of processes for the current node.

    Returns:
        int: The local world size.
    """
    return _get_distributed_config_var(env_var="LOCAL_WORLD_SIZE", default=1, human_name="local world size")


def get_local_rank() -> int:
    """Returns the local rank for the current process, which is on ``[0; LOCAL_WORLD_SIZE - 1]``.

    Returns:
        int: The local rank.
    """
    return _get_distributed_config_var(env_var="LOCAL_RANK", default=0, human_name="local rank")


def get_node_rank() -> int:
    """Returns the node rank. For example, if there are 2 nodes, and 2 ranks per node, then global ranks 0-1 will have a
    node rank of 0, and global ranks 2-3 will have a node rank of 1.

    Returns:
        int: The node rank, starting at 0.
    """
    return _get_distributed_config_var(env_var="NODE_RANK", default=0, human_name="node rank")


def barrier() -> None:
    """Synchronizes all processes.

    This function blocks until all processes reach this function.

    .. seealso:: :func:`torch.distributed.barrier`
    """
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
    """Reduce a ``tensor`` by applying the ``reduce_operation``.

    All ranks get the same, bitwise-identical result.

    .. seealso:: :func:`torch.distributed.all_reduce`

    Args:
        tensor (torch.Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
    Args:
        tensor (torch.Tensor): Tensor to reduce. The function operates in-place.
        reduce_operation (str, optional): The reduction operation (default: ``SUM``).

            Valid options are:
                * ``SUM``
                * ``PRODUCT``
                * ``MIN``
                * ``MAX``
                * ``BAND``
                * ``BOR``
                * ``BXOR``

    Returns:
        None: ``tensor`` is modified in-place.
    """
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
    See :func:`torch.distributed.broadcast`.

    Args:
        tensor (torch.Tensor): Data to be sent if ``src`` is the rank of current process,
            and tensor to be used to save received data otherwise.
        src (int): Source rank
    """
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src)
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(f"The world_size({world_size}) > 1, but the distributed package is not "
                       "available or has not been initialized. Please check you have initialized "
                       "the distributed runtime and that PyTorch has been built with distributed "
                       "support.")


def broadcast_object_list(object_list: List[Any], src: int = 0) -> None:
    """Broadcasts picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be broadcasted.

    .. seealso:: :func:`torch.distributed.broadcast`.

    Args:
        object_list (torch.Tensor): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will be broadcast,
            but each rank must provide lists of equal sizes.
        src (int, optional): Source rank (default: ``0``)
    Returns:
        None:  ``object_list`` will be modified in-place and set to values of ``object_list`` from the ``src`` rank.
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
    """Collects a :class:`~torch.Tensor` from each rank and return a sequence of
    :class:`~torch.Tensor`\\s indexed by rank.

    .. seealso:: :func:`torch.distributed.all_gather`

    Args:
        tensor (torch.Tensor): Tensor from each rank to be gathered.

    Returns:
        Sequence[Tensor]: A sequence of tensors indexed by rank.
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
    """Collect a pickleable object from each rank and return a list of these objects indexed by rank.

    .. seealso:: :func:`torch.distributed.all_gather_object`

    Args:
        obj (TObj): Object to be gathered.

    Returns:
        List[TObj]: A list of objects indexed by rank.
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
    """Returns whether PyTorch was built with distributed support.

    .. seealso:: :func:`torch.distributed.is_available`

    Returns:
        bool: Whether PyTorch distributed support is available.
    """
    return dist.is_available()


def is_initialized():
    """Returns whether PyTorch distributed is initialized.

    .. seealso:: :func:`torch.distributed.is_initialized`

    Returns:
        bool: Whether PyTorch distributed is initialized.
    """
    return dist.is_initialized()


def initialize_dist(backend: str, timeout: datetime.timedelta):
    """Initialize the default PyTorch distributed process group.

    This function assumes that the following environment variables are set:

    * ``RANK``: The global rank of the process, which should be on ``[0; WORLD_SIZE - 1]``.
    * ``LOCAL_RANK``: The local rank for the process, which should be on ``[0; LOCAL_WORLD_SIZE - 1]``.
    * ``NODE_RANK``: The rank of the node.
    * ``WORLD_SIZE``: The total number of processes.
    * ``LOCAL_WORLD_SIZE``: The number of processes on the current node.
    * ``MASTER_ADDR``: The hostname for the rank-zero process.
    * ``MASTER_PORT``: The port for the rank-zero process.

    If none of the environment variables are set, this function will assume a single-rank
    configuration and initialize the default process group using a :class:`torch.distributed.HashStore` store.

    .. seealso:: :func:`torch.distributed.init_process_group`

    Args:
        backend (str): The distributed backend to use. Should be ``gloo`` for CPU training,
            or ``nccl`` for GPU training.
        timeout (datetime.timedelta): The timeout for operations exected against the process group.
    """
    if get_world_size() == 1:
        warnings.warn("DistributedWarning: Initializing of torch.distributed required but the world size is 1."
                      "This is supported, but not recommended.")

    if get_world_size() > 1 and not dist.is_available():
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

    dist_env_variable_names = ("NODE_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK")

    is_missing_all_dist_env_vars = all(x not in os.environ for x in dist_env_variable_names)
    if is_missing_all_dist_env_vars:
        # missing all variables, in which case we should assume a single process
        # if any variables are set, then it's likely an incomplete configuration, in which case we should not assume
        # defaults (it would be better to let dist.init_process_group crash)
        warnings.warn(
            textwrap.dedent(f"""\
                NoDistributedWarning: No distributed environment variables are set; assuming no
                parallelization. If this is unexpected, please run the script with the composer CLI tool."""))
        # setting the environment variables to single-rank defaults
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["NODE_RANK"] = "0"
        dist.init_process_group(backend, store=dist.HashStore(), world_size=1, rank=0)
        return

    dist.init_process_group(backend, timeout=timeout)


def get_sampler(dataset: torch.utils.data.Dataset, *, drop_last: bool, shuffle: bool):
    """Constructs a :class:`~torch.utils.data.distributed.DistributedSampler` for a dataset. The
    :class:`~torch.utils.data.distributed.DistributedSampler` assumes that each rank has a complete copy of the dataset.
    It ensures that each rank sees a unique shard for each epoch containing ``len(datset) / get_world_size()`` samples.

    .. note::

        If the ``dataset`` is already shareded by rank, use a :class:`~torch.utils.data.SequentialSampler`
        or :class:`~torch.utils.data.RandomSampler`.

    Args:
        dataset (torch.utils.data.Dataset): The dataset.
        drop_last (bool): Whether to trop the last batch.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        torch.utils.data.distributed.DistributedSampler: The sampler.
    """
    return torch.utils.data.DistributedSampler[int](
        dataset,
        drop_last=drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
    )
