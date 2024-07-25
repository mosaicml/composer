# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The base :class:`~composer.devices.device.Device` class."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Callable, TypeVar

import torch
import torch.nn
from torch.optim import Optimizer

__all__ = ['Device', 'T_nnModule']

T_nnModule = TypeVar('T_nnModule', bound=torch.nn.Module)
T_Batch = TypeVar('T_Batch')


class Device(ABC):
    """Abstract class for a device on which a model runs.

    Attributes:
        dist_backend (str): Distributed backend to use.
            Should be ``gloo``, ``mpi``, or ``nccl``.
            See `the pytorch docs <https://pytorch.org/docs/stable/distributed.html>`_
            for details.
    """

    dist_backend: str = ''
    name: str = ''
    _device = None

    @abstractmethod
    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        """Invoked by the :class:`.Trainer` to move a ``module`` onto the device.

        Args:
            module (torch.nn.Module): The module to move to the device.

        Returns:
            torch.nn.Module: The module on the device.
        """
        pass

    @abstractmethod
    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Invoked by the :class:`.Trainer` to move a tensor onto a device.

        Args:
            tensor (Tensor): The tensor to move to the device.

        Returns:
            Tensor: The tensor on the device.
        """
        pass

    def batch_to_device(self, batch: T_Batch) -> T_Batch:
        """Invoked by the :class:`.Trainer` move all tensors items in a batch to device.

        Supports nested sequences and mappings of tensors. Ignores non-tensor items. Preserves sequence and mapping types
        when possible; otherwise, sequences are converted to lists, and mappings are converted to dictionaries.

        Args:
            batch (Any): The batch to move to the device.

        Returns:
            Batch: The batch on the device.
        """

        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return self.tensor_to_device(x)
            return x

        return _map_batch(batch, _to_device)

    def optimizer_to_device(self, optimizer: Optimizer) -> Optimizer:
        """Invoked by the :class:`.Trainer` to move the optimizer's state onto the device.

        Args:
            optimizer (Optimizer): The optimizer to move to the device

        Returns:
            Optimizer: The optimizer on the device
        """
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = self.tensor_to_device(v)
        return optimizer


def _map_batch(batch: Any, map_fn: Callable) -> Any:
    """Recursively maps a function to all items in a batch.

    Args:
        batch: Nested lists and dictionaries.
        map_fn: A function to invoke on each element.

    Returns:
        Collections: The result of applying ``map_fn`` on each element of the ``batch``.
        The type of ``batch`` is preserved.
    """
    if isinstance(batch, Mapping):
        return {k: _map_batch(v, map_fn) for k, v in batch.items()}

    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        try:
            return type(batch)(_map_batch(x, map_fn) for x in batch)  # type: ignore
        except TypeError:
            return [_map_batch(x, map_fn) for x in batch]
    return map_fn(batch)
