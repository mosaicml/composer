# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`World` class is used for easily querying distributed training info, used by `StreamingDataset`.
"""

from typing import NamedTuple

from torch.utils.data import get_worker_info

from composer.utils import dist

__all__ = ['World', 'get_world']


class World(NamedTuple):
    """A :class:`NamedTuple` that provides context about workers, devices, and nodes.


    Attributes:
        global_node (int): The id of this node within the global system
        global_num_nodes (int): The number of nodes within the global system
        global_device (int): The id of this device within the global system
        global_num_devices (int): The number of devices within the global system
        node_device (int): The id of this device within this node
        node_num_devices (int): The number of devices within this node
        global_worker (int): The id of this worker within the global system
        global_num_workers (int): The number of workers within the global system
        node_worker (int): The id of this worker within this node
        node_num_workers (int): The number of workers within this node
        device_worker (int): The id of this worker within this device
        device_num_workers (int): The number of workers within this device
    """
    global_node: int
    global_num_nodes: int

    global_device: int
    global_num_devices: int

    node_device: int
    node_num_devices: int

    global_worker: int
    global_num_workers: int

    node_worker: int
    node_num_workers: int

    device_worker: int
    device_num_workers: int


def get_world() -> World:
    """Returns a :class:`World` object, initialized using :mod:`composer.utils.dist` and :func:`torch.utils.data.get_worker_info`"""
    # Node and Device info
    global_node = dist.get_node_rank()
    global_device = dist.get_global_rank()
    global_num_devices = dist.get_world_size()
    node_device = dist.get_local_rank()
    node_num_devices = dist.get_local_world_size()

    # TODO: to remove this block, composer.dist must provide 'num_nodes'
    if global_num_devices % node_num_devices != 0:
        raise RuntimeError(
            f"Expected global_num_devices ({global_num_devices}) % node_num_devices ({node_num_devices}) == 0. Unable to determine 'num_nodes'."
        )
    global_num_nodes = global_num_devices // node_num_devices

    # Worker info
    # We assume every Device has the same number of Workers.
    worker_info = get_worker_info()
    if worker_info:
        device_worker = worker_info.id
        device_num_workers = worker_info.num_workers
    else:
        device_worker = 0
        device_num_workers = 1

    node_worker = node_device * device_num_workers + device_worker
    node_num_workers = node_num_devices * device_num_workers

    global_worker = global_device * device_num_workers + device_worker
    global_num_workers = global_num_devices * device_num_workers

    return World(
        global_node=global_node,
        global_num_nodes=global_num_nodes,
        global_device=global_device,
        global_num_devices=global_num_devices,
        node_device=node_device,
        node_num_devices=node_num_devices,
        global_worker=global_worker,
        global_num_workers=global_num_workers,
        node_worker=node_worker,
        node_num_workers=node_num_workers,
        device_worker=device_worker,
        device_num_workers=device_num_workers,
    )
