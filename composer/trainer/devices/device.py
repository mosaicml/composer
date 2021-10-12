# Copyright 2021 MosaicML. All Rights Reserved.

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator, Optional, TypeVar, Union

import torch.nn

from composer.core.serializable import Serializable
from composer.core.state import State
from composer.core.types import DataLoader, Optimizer, Precision, Tensor, TPrefetchFn

T_nnModule = TypeVar("T_nnModule", bound=torch.nn.Module)


class Device(Serializable, ABC):

    @abstractmethod
    def prepare(self, state: State) -> None:
        """prepare() is invoked by the trainer at the beginning of the training loop.
        The device should perform any initialization here.
        It should not modify the state.

        Args:
            state (State): The global state variable.
        """

    @abstractmethod
    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        """module_to_device moves a module onto the device's device.

        Args:
            module (T_nnModule): The module to move to the device

        Returns:
            T_nnModule: The module on the device.
        """
        pass

    @abstractmethod
    def tensor_to_device(self, tensor: Tensor) -> Tensor:
        """tensor_to_device moves a tensor onto the device's device.

        Args:
            tensor (T_nnModule): The tensor to move to the device

        Returns:
            Tensor: The tensor on the device.
        """
        pass

    @abstractmethod
    def dataloader_to_device(self, dataloader: DataLoader, prefetch_fn: Optional[TPrefetchFn]) -> DataLoader:
        """dataloader_to_device wraps a dataloader and ensures that all returned batches are on the
        the correct device. It is responsible for executing `prefetch_fn`, if provided, on each batch before it is
        yielded. The `prefetch_fn` can be executed in the background, if the device supports it.

        Args:
            dataloader (DataLoader): The dataloader to wrap.
            prefetch_fn (Optional[TPrefetchFn]): A function that takes a batch and returns a batch.
                It should perform any on-device preprocessing of a batch.
                (e.g. on a GPU device, this function can be used for gpu transformations.)

        Returns:
            DataLoader: The wrapped dataloader, which yields batches that have been moved to the device
                and have been processed through the prefetch_fn.
        """

    def optimizer_to_device(self, optimizer: Optimizer) -> Optimizer:
        """optimizer_to_device moves an optimizer's state onto the device's device.

        As a rule, this usually isn't necessary, since most optimizers lazy initialize their state
        when `.step()` is first called, based off of the device of the parameters. The prominent
        exception to this rule is when we are restoring from a checkpoint.

        Args:
            optimizer (Optimizer): The optimizer to move to the device

        Returns:
            Optimizer: The optimizer on the device
        """
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = self.tensor_to_device(v)
        return optimizer

    @abstractmethod
    @contextmanager
    def precision_context(self, precision: Union[str, Precision]) -> Generator[None, None, None]:
        """precision returns a context manager that uses the specified precision.

        Example usage:

        ```
        with device.precision(Precision.AMP):
            forward_pass_with_amp()
        ```

        Args:
            precision (Precision): [description]

        Yields:
            Generator[None, None, None]: [description]
        """
        pass

    @property
    @abstractmethod
    def nproc_per_node(self) -> int:
        """nproc_per_node returns the number of processes per node the device would like to have.
        This function may be invoked before `prepare()` and therefore should not rely on state.

        Returns:
            int: Number of processes per node.
        """

    @property
    @abstractmethod
    def ddp_backend(self) -> str:
        """DDP backend to use. Should return `gloo`, `mpi`, or `nccl`.
        See https://pytorch.org/docs/stable/distributed.html

        Returns:
            str: `gloo`, `mpi`, or `nccl`
        """
