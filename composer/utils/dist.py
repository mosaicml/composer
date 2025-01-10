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
import io
import logging
import os
import pickle
import random
import string
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Sequence, TypeVar, Union, cast

import torch
import torch.distributed as dist
import torch.utils.data

from composer.utils.device import get_device, is_hpu_installed

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

_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


def _object_to_tensor(obj, device):
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size


def _tensor_to_object(tensor, tensor_size):
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()


def all_gather_object_list_hpu(object_list, obj, group=None):
    """Use this only for habana devices, for other devices use all_gather_object.

    Function is a modified version of
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py.
    Gathers picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the object
    must be picklable in order to be gathered.

    Args:
        object_list (list[Any]): Output list. It should be correctly sized as the
            size of the group for this collective and will contain the output.
        obj (Any): Pickable Python object to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list`` will
        be unmodified.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        :func:`all_gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.all_gather_object(output, gather_objects[dist.get_rank()])
        >>> output
        ['foo', 12, {1: 2}]
    """
    current_device = torch.device('hpu')
    input_tensor, local_size = _object_to_tensor(obj, current_device)
    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = dist.get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size, dtype=torch.long, device=current_device)
    object_size_list = [object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)]
    # Allgather tensor sizes
    dist.all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(max_object_size * group_size, dtype=torch.bfloat16, device=current_device)
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [coalesced_output_tensor[max_object_size * i:max_object_size * (i + 1)] for i in range(group_size)]
    dist.all_gather(output_tensors, input_tensor.to(torch.bfloat16), group=group)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)


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
                raise RuntimeError(
                    'Torch distributed has been initialized with a value of '
                    f'{dist_value} for {human_name}, but environment variable '
                    f'{env_var} has value {env_value}.',
                )
        return dist_value

    if env_var in os.environ:
        return int(os.environ[env_var])

    if dist.is_initialized():
        raise MissingEnvironmentError(
            'Torch distributed is initialized but environment variable '
            f'{env_var} is not set.',
        )

    return default


def get_world_size() -> int:
    """Returns the world size, which is the number of processes participating in this training run.

    Returns:
        int: The world size.
    """
    return _get_distributed_config_var(
        env_var='WORLD_SIZE',
        human_name='world size',
        default=1,
        fetch_fn_name='get_world_size',
    )


def get_global_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Returns the global rank of the current process in the input PG, which is on ``[0; group.WORLD_SIZE - 1]``.

    Args:
        group (ProcessGroup, optional): The process group. If ``None``, it will return env_var ``RANK``

    Returns:
        int: The global rank in input process group.
    """
    if group is None:
        return _get_distributed_config_var(
            env_var='RANK',
            human_name='global rank',
            default=0,
            fetch_fn_name='get_rank',
        )
    return dist.get_rank(group)


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


def barrier(group=None) -> None:
    """Synchronizes all processes.

    This function blocks until all processes reach this function.

    .. seealso:: :func:`torch.distributed.barrier`

    Args:
        group (ProcessGroup, optional): The process group to work on. If ``None``,
            the default process group will be used. Default is ``None``.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=group)
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(
        f'The world_size({world_size}) > 1, but the distributed package is not '
        'available or has not been initialized. Please check you have initialized '
        'the distributed runtime and that PyTorch has been built with distributed '
        'support. If calling this function outside Trainer, please ensure that '
        '`composer.utils.dist.initialize_dist` has been called first.',
    )


def all_reduce(
    tensor: torch.Tensor,
    reduce_operation: str = 'SUM',
    group=None,
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
        group (ProcessGroup, optional): The process group to work on. If ``None``,
            the default process group will be used. Default is ``None``.
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
        dist.all_reduce(tensor, op=reduce_op, group=group)
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(
        f'The world_size({world_size}) > 1, but the distributed package is not '
        'available or has not been initialized. Please check you have initialized '
        'the distributed runtime and that PyTorch has been built with distributed '
        'support. If calling this function outside Trainer, please ensure that '
        '`composer.utils.dist.initialize_dist` has been called first.',
    )


def broadcast(tensor: torch.Tensor, src: int, group=None) -> None:
    """Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes participating in the collective.
    See :func:`torch.distributed.broadcast`.

    Args:
        tensor (torch.Tensor): Data to be sent if ``src`` is the rank of current process,
            and tensor to be used to save received data otherwise.
        src (int): Source rank
        group (ProcessGroup, optional): The process group to work on. If ``None``,
            the default process group will be used. Default is ``None``.
    """
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src=src, group=group)
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(
        f'The world_size({world_size}) > 1, but the distributed package is not '
        'available or has not been initialized. Please check you have initialized '
        'the distributed runtime and that PyTorch has been built with distributed '
        'support. If calling this function outside Trainer, please ensure that '
        '`composer.utils.dist.initialize_dist` has been called first.',
    )


def broadcast_object_list(object_list: list[Any], src: int = 0, group=None) -> None:
    """Broadcasts picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be broadcasted.

    .. seealso:: :func:`torch.distributed.broadcast`.

    Args:
        object_list (torch.Tensor): list of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will be broadcast,
            but each rank must provide lists of equal sizes.
        src (int, optional): Source rank (default: ``0``)
        group (ProcessGroup, optional): The process group to work on. If ``None``,
            the default process group will be used. Default is ``None``.

    Returns:
        None:  ``object_list`` will be modified in-place and set to values of ``object_list`` from the ``src`` rank.
    """
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(object_list, src=src, group=group)
        # torch.distributed will replace the None's in obj_gather_list with the gathered objects on rank 0
        # or will just be None on non-rank-0
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(
        f'The world_size({world_size}) > 1, but the distributed package is not '
        'available or has not been initialized. Please check you have initialized '
        'the distributed runtime and that PyTorch has been built with distributed '
        'support. If calling this function outside Trainer, please ensure that '
        '`composer.utils.dist.initialize_dist` has been called first.',
    )


def all_gather(tensor: torch.Tensor, group=None) -> Sequence[torch.Tensor]:
    """Collects a :class:`~torch.Tensor` from each rank.

    .. seealso:: :func:`torch.distributed.all_gather`

    Args:
        tensor (torch.Tensor): Tensor from each rank to be gathered.
        group (ProcessGroup, optional): The process group to work on. If ``None``,
            the default process group will be used. Default is ``None``.

    Returns:
        Sequence[Tensor]: A sequence of tensors indexed by rank.
    """
    if dist.is_available() and dist.is_initialized():
        obj_gather_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        dist.all_gather(obj_gather_list, tensor, group=group)
        return obj_gather_list
    world_size = get_world_size()
    if world_size == 1:
        return [tensor]
    raise RuntimeError(
        f'The world_size({world_size}) > 1, but the distributed package is not '
        'available or has not been initialized. Please check you have initialized '
        'the distributed runtime and that PyTorch has been built with distributed '
        'support. If calling this function outside Trainer, please ensure that '
        '`composer.utils.dist.initialize_dist` has been called first.',
    )


def all_gather_object(obj: TObj, group=None) -> list[TObj]:
    """Collect a pickleable object from each rank and return a list of these objects indexed by rank.

    .. seealso:: :func:`torch.distributed.all_gather_object`

    Args:
        obj (TObj): Object to be gathered.
        group (ProcessGroup, optional): The process group to work on. If ``None``,
            the default process group will be used. Default is ``None``.

    Returns:
        list[TObj]: A list of objects indexed by rank.
    """
    if dist.is_available() and dist.is_initialized():
        obj_gather_list = [None for _ in range(get_world_size())]
        if is_hpu_installed():
            all_gather_object_list_hpu(obj_gather_list, obj, group=group)
        else:
            dist.all_gather_object(obj_gather_list, obj, group=group)
        # torch.distributed will replace the None's in obj_gather_list with the gathered objects on rank 0
        # or will just be None on non-rank-0
        return cast(list[TObj], obj_gather_list)
    world_size = get_world_size()
    if world_size == 1:
        return [obj]
    raise RuntimeError(
        f'The world_size({world_size}) > 1, but the distributed package is not '
        'available or has not been initialized. Please check you have initialized '
        'the distributed runtime and that PyTorch has been built with distributed '
        'support. If calling this function outside Trainer, please ensure that '
        '`composer.utils.dist.initialize_dist` has been called first.',
    )


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


def initialize_dist(device: Optional[Union[str, Device]] = None, timeout: float = 300.0) -> None:
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
        device (Optional[str | Device] ): The device from which the distributed backend is
            interpreted. Either a string corresponding to a device (one of ``'cpu'``,
            ``'gpu'``, ``'mps'``, or ``'tpu'``) or a :class:`.Device`. (default: ``None``)
        timeout (float, optional): The timeout for operations executed against the process
            group, expressed in seconds. (default: ``300.0``).
    """
    # If device is string, get corresponding composer.devices.Device object
    device_obj = get_device(device)
    timeout_timedelta = datetime.timedelta(seconds=timeout)

    if get_world_size() > 1 and not dist.is_available():
        raise RuntimeError(
            'When the world size is > 1, ``torch.distributed`` must be used. However, it is '
            'not available in your installation of PyTorch. Please install or build PyTorch '
            'with distributed support.',
        )

    if dist.is_initialized():
        if dist.get_backend() != device_obj.dist_backend.lower():
            raise RuntimeError(
                f'The requested backend ({device_obj.dist_backend}) differs from the backend '
                f'of the current process group ({dist.get_backend()}). If you '
                'wish to change backends, please restart the python process.',
            )
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

    if device_obj.dist_backend == 'xla':
        if not 'torch_xla' in sys.modules:
            raise RuntimeError(
                'PyTorch XLA package not found. In order to use XLA based devices '
                'PyTorch XLA must be installed.',
            )
        # XLA initialization requires the init_method to be set
        dist.init_process_group(device_obj.dist_backend, init_method='xla://')
    elif dist_env_vars_match_defaults:
        # Fill in the remaining single-rank variables
        os.environ.update(dist_env_var_defaults)
        dist.init_process_group(device_obj.dist_backend, store=dist.HashStore(), world_size=1, rank=0)
    else:
        dist.init_process_group(device_obj.dist_backend, timeout=timeout_timedelta)


def get_sampler(
    dataset: torch.utils.data.Dataset,
    *,
    drop_last: bool = False,
    shuffle: bool = False,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    seed: int = 0,
):
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
        num_replicas (int, optional): The number of replicas. If ``None``, defaults to the world size.
        rank (int, optional): The rank. If ``None``, defaults to the global rank.

    Returns:
        torch.utils.data.distributed.DistributedSampler: The sampler.
    """
    return torch.utils.data.DistributedSampler[int](
        dataset,
        drop_last=drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size() if num_replicas is None else num_replicas,
        rank=get_global_rank() if rank is None else rank,
        seed=seed,
    )


def get_node_signal_file_name(rng: Optional[random.Random] = None) -> str:
    """Returns a file name to use for a file based wait within a node.

    The file name will contain a randomly generated string to avoid conflicts.
    Note: This file name will be the same on each node, so that it can be used for a file based wait.

    Returns:
        str: The name of the file that will be created to signal the end of a node's training.
    """
    if rng is None:
        rng = random.Random()

    random_string = ''.join(rng.choices(string.ascii_letters + string.digits, k=6))
    node_rank = get_node_rank()
    file_name_list = [f'._signal_file_node{node_rank}_{random_string}']
    broadcast_object_list(file_name_list, src=0)
    return file_name_list[0]


def write_signal_file(signal_file_name: str, dir_path: Optional[str] = None) -> str:
    """Writes a signal file to the specified directory.

    This function creates a signal file in the specified directory. If the directory does
    Note: Only local rank zero writes the signal file. All other ranks are expected to wait for the signal file.

    Args:
        signal_file_name (str): The name of the signal file.
        dir_path (str, optional): The full path to the directory in which to create the signal file. If ``None``,
            the current working directory will be used.
    """
    if dir_path is not None:
        os.makedirs(dir_path, exist_ok=True)

    signal_file_path = os.path.join(dir_path or os.getcwd(), signal_file_name)
    if get_local_rank() == 0:
        with open(signal_file_path, 'w') as _f:
            _f.write('local rank zero done')

    return signal_file_path


@contextmanager
def busy_wait_for_local_rank_zero(dir_path: Optional[str] = None):
    """Busy waits for the signal file to be created by local rank zero.

    This function will wait for the signal file to be created by local rank zero. It will
    check every 0.1 seconds for the existence of the file.

    Args:
        dir_path (str, optional): The directory in which to look for the signal file. If ``None``,
            the current working directory will be used.
    """
    # Get unique file name
    signal_file_name = get_node_signal_file_name()

    # All ranks yield execution to allow local rank zero to run the code it needs to
    yield

    # Local rank zero writes the signal file, all other rank just get the expected path
    signal_file_path = write_signal_file(signal_file_name=signal_file_name, dir_path=dir_path)

    # Wait for the signal file to be created by local rank zero
    with local_rank_zero_download_and_wait(signal_file_path):
        # Sync all ranks across nodes as busy wait only is within node
        dist.barrier()

    # Remove the signal file
    if get_local_rank() == 0:
        os.remove(signal_file_path)


@contextmanager
def local_rank_zero_download_and_wait(expected_file_path: str):
    """Context manager to wait for a file to exist on all ranks except local rank zero.

    It is expected that the file will be created by local rank zero. This function is useful
    as an alternative to ``run_local_rank_zero_first`` when downloading a file, because it does
    not require dist to be initialized. It only requires that the ``LOCAL_RANK`` environment variable
    is set. If dist is initialized, you should use ``run_local_rank_zero_first`` instead to avoid busy waiting.

    Args:
        expected_file_path (str): The file to wait for existence of
    """
    local_rank = get_local_rank()
    if local_rank != 0:
        while not os.path.exists(expected_file_path):
            time.sleep(0.1)

    yield


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
    raise RuntimeError(
        f'The world_size({world_size}) > 1, but the distributed package is not '
        'available or has not been initialized. Please check you have initialized '
        'the distributed runtime and that PyTorch has been built with distributed '
        'support. If calling this function outside Trainer, please ensure that '
        '`composer.utils.dist.initialize_dist` has been called first.',
    )
