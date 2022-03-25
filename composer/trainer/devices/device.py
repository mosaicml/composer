# Copyright 2021 MosaicML. All Rights Reserved.

"""The base :class:`~composer.trainer.devices.device.Device` class."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar, Union

import torch
import torch.nn
from torch._six import string_classes
from torch.optim import Optimizer

from composer.core.precision import Precision
from composer.core.serializable import Serializable

__all__ = ["Device", "T_nnModule"]

T_nnModule = TypeVar("T_nnModule", bound=torch.nn.Module)
T_Batch = TypeVar('T_Batch')


class Device(Serializable, ABC):
    """Abstract class for a device on which a model runs.

    Attributes:
        dist_backend (str): Distributed backend to use.
            Should be ``gloo``, ``mpi``, or ``nccl``.
            See `the pytorch docs <https://pytorch.org/docs/stable/distributed.html>`_
            for details.
    """

    dist_backend: str

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
        """Invoked by the :class:`.Trainer` to move the ``batch`` onto the device.

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

    @abstractmethod
    @contextmanager
    def precision_context(self, precision: Union[str, Precision]) -> Generator[None, None, None]:
        """Precision returns a context manager that uses the specified precision.

        Example usage:

        .. doctest::

            >>> from composer.core.precision import Precision
            >>> from composer.trainer.devices import DeviceCPU
            >>>
            >>> device = DeviceCPU()
            >>> for batch in train_dataloader:
            ...     with device.precision_context(Precision.FP32):
            ...         outputs = model.forward(batch)
            ...
            ...     with device.precision_context(Precision.FP32):
            ...         loss = model.loss(outputs, batch)
            >>>

        Args:
            precision (Precision): The desired precision for the device.

        Yields:
            Generator[None, None, None]: A context for the precision.
        """
        pass


def _map_batch(batch: Any, map_fn: Callable) -> Any:
    """Recursively maps a function to the values of nested lists and dictionaries.

    Args:
        batch: Nested lists and dictionaries.
        map_fn: A function to invoke on each element.

    Returns:
        Collections: The result of applying ``map_fn`` on each element of the ``batch``.
        The type of ``batch`` is preserved.
    """
    if isinstance(batch, torch.Tensor):
        return map_fn(batch)
    if isinstance(batch, Mapping):
        return {k: _map_batch(v, map_fn) for k, v in batch.items()}
    elif isinstance(batch, Sequence) and not isinstance(batch, string_classes):
        try:
            return type(batch)(_map_batch(x, map_fn) for x in batch)  # type: ignore
        except TypeError:
            return [_map_batch(x, map_fn) for x in batch]
    else:
        return map_fn(batch)
