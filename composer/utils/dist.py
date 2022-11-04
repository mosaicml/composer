# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

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
import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, TypeVar, Union, cast

import torch
import torch.distributed as dist
import torch.utils.data

from composer.utils.device import get_device

if TYPE_CHECKING:
    from composer.devices import Device

TObj = TypeVar('TObj')

__all__ = [
    'all_gather',
    'all_gather_object',
    'all_reduce',
    'barrier',
    'broadcast',
    'broadcast_object_list',
    'get_global_rank',
    'get_local_rank',
    'get_local_world_size',
    'get_node_rank',
    'get_sampler',
    'get_world_size',
    'initialize_dist',
    'is_available',
    'is_initialized',
]

log = logging.getLogger(__name__)


class MissingEnvironmentError(Exception):
    pass


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
                raise RuntimeError('Torch distributed has been initialized with a value of '
                                   f'{dist_value} for {human_name}, but environment variable '
                                   f'{env_var} has value {env_value}.')
        return dist_value

    if env_var in os.environ:
        return int(os.environ[env_var])

    if dist.is_initialized():
        raise MissingEnvironmentError('Torch distributed is initialized but environment variable '
                                      f'{env_var} is not set.')

    return default


def get_world_size() -> int:
    """Returns the world size, which is the number of processes participating in this training run.

    Returns:
        int: The world size.
    """
    return _get_distributed_config_var(env_var='WORLD_SIZE',
                                       human_name='world size',
                                       default=1,
                                       fetch_fn_name='get_world_size')


def get_global_rank() -> int:
    """Returns the global rank of the current process, which is on ``[0; WORLD_SIZE - 1]``.

    Returns:
        int: The global rank.
    """
    return _get_distributed_config_var(env_var='RANK', human_name='global rank', default=0, fetch_fn_name='get_rank')


def get_local_world_size() -> int:
    """Returns the local world size, which is the number of processes for the current node.

    Returns:
        int: The local world size.
    """
    return _get_distributed_config_var(env_var='LOCAL_WORLD_SIZE', default=1, human_name='local world size')


def get_local_rank() -> int:
    """Returns the local rank for the current process, which is on ``[0; LOCAL_WORLD_SIZE - 1]``.

    Returns:
        int: The local rank.
    """
    return _get_distributed_config_var(env_var='LOCAL_RANK', default=0, human_name='local rank')


def get_node_rank() -> int:
    """Returns the node rank.

    For example, if there are 2 nodes, and 2 ranks per node, then global ranks 0-1 will have a
    node rank of 0, and global ranks 2-3 will have a node rank of 1.

    Returns:
        int: The node rank, starting at 0.
    """
    return _get_distributed_config_var(env_var='NODE_RANK', default=0, human_name='node rank')


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
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


def all_reduce(
    tensor: torch.Tensor,
    reduce_operation: str = 'SUM',
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
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


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
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


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
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


def all_gather(tensor: torch.Tensor) -> Sequence[torch.Tensor]:
    """Collects a :class:`~torch.Tensor` from each rank.

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
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


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
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')


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


def initialize_dist(device: Union[str, Device], timeout: float = 300.0):
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
        device (str | Device): The device from which the distributed backend is
            interpreted. Either a string corresponding to a device (one of ``'cpu'``,
            ``'gpu'``, ``'mps'``, or ``'tpu'``) or a :class:`.Device`.
        timeout (float, optional): The timeout for operations executed against the process
            group, expressed in seconds. (default: ``300.0``).
    """
    # If device is string, get corresponding composer.devices.Device object
    device_obj = get_device(device)
    timeout_timedelta = datetime.timedelta(seconds=timeout)

    if get_world_size() > 1 and not dist.is_available():
        raise RuntimeError('When the world size is > 1, ``torch.distributed`` must be used. However, it is '
                           'not available in your installation of PyTorch. Please install or build PyTorch '
                           'with distributed support.')

    if dist.is_initialized():
        if dist.get_backend() != device_obj.dist_backend.lower():
            raise RuntimeError(f'The requested backend ({device_obj.dist_backend}) differs from the backend '
                               f'of the current process group ({dist.get_backend()}). If you '
                               'wish to change backends, please restart the python process.')
        return

    # If any of these variables are set, and they do not match the single rank defaults,
    # then do not automatically configure distributed. There are no reasonable defaults to infer
    # for the other variables. Instead, let torch.dist error on an incomplete configuration.

    # If none of these variables are set, or some are set but they match the single rank defaults,
    # then fill the rest in.

    dist_env_var_defaults = {
        'NODE_RANK': '0',
        'WORLD_SIZE': '1',
        'LOCAL_WORLD_SIZE': '1',
        'RANK': '0',
        'LOCAL_RANK': '0',
    }

    log.debug(
        'Initializing torch.dist: global_rank=%d, local_rank=%d, world_size=%d, local_world_size=%d, node_rank=%d',
        get_global_rank(),
        get_local_rank(),
        get_world_size(),
        get_local_world_size(),
        get_node_rank(),
    )

    dist_env_vars_match_defaults = all(os.environ.get(k, v) == v for (k, v) in dist_env_var_defaults.items())

    if dist_env_vars_match_defaults:
        # Fill in the remaining single-rank variables
        os.environ.update(dist_env_var_defaults)
        dist.init_process_group(device_obj.dist_backend, store=dist.HashStore(), world_size=1, rank=0)
    else:
        dist.init_process_group(device_obj.dist_backend, timeout=timeout_timedelta)


def get_sampler(dataset: torch.utils.data.Dataset, *, drop_last: bool = False, shuffle: bool = False):
    """Constructs a :class:`~torch.utils.data.distributed.DistributedSampler` for a dataset.

    The :class:`~torch.utils.data.distributed.DistributedSampler` assumes that each rank has a complete copy of the
    dataset. It ensures that each rank sees a unique shard for each epoch containing
    ``len(dataset) / get_world_size()`` samples.

    .. note::

        If the ``dataset`` is already sharded by rank, use a :class:`~torch.utils.data.SequentialSampler`
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


@contextmanager
def run_local_rank_zero_first():
    """Context manager to hold all non-zero ranks until rank zero completes.

    The below example will let the local rank zero download
    the dataset, and hold all non-rank zeros until the
    download is complete.

    .. code-block: python

        with run_local_rank_zero_first():
            dataset = CIFAR10(
                ...,
                download=True,
            )

    This prevents race conditions where multiple
    ranks attempt to download the dataset to the
    same location.
    """
    if dist.is_available() and dist.is_initialized():
        # hold non-zero ranks until rank zero done
        if get_local_rank() != 0:
            dist.barrier()
            yield
        else:
            yield
            dist.barrier()
        return
    world_size = get_world_size()
    if world_size == 1:
        yield
        return
    raise RuntimeError(f'The world_size({world_size}) > 1, but the distributed package is not '
                       'available or has not been initialized. Please check you have initialized '
                       'the distributed runtime and that PyTorch has been built with distributed '
                       'support. If calling this function outside Trainer, please ensure that '
                       '`composer.utils.dist.initialize_dist` has been called first.')
