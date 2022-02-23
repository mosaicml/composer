# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed
import torch.utils.data
import yahp as hp

from composer.core.types import Batch, DataLoader, Dataset

log = logging.getLogger(__name__)


class WrappedDataLoader(DataLoader):

    def __init__(self, dataloader: DataLoader) -> None:
        if self.is_dataloader_already_wrapped(dataloader):
            log.debug(
                textwrap.dedent("""\
                    The dataloader is already wrapped with %s; it will be wrapped again.
                    If this is unintended behavior, guard the wrapping of the dataloader i.e. with:
                    if not %s.is_dataloader_already_wrapped(dataloader): dataloader = %s(dataloader)"""),
                type(self).__name__,
                type(self).__name__,
                type(self).__name__,
            )
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self.num_workers = dataloader.num_workers
        self.pin_memory = dataloader.pin_memory
        self.drop_last = dataloader.drop_last
        self.timeout = dataloader.timeout
        self.sampler = dataloader.sampler
        self.prefetch_factor = dataloader.prefetch_factor
        self.dataloader = dataloader

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.dataloader)

    def __bool__(self) -> bool:
        return True

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name) and name in ("dataset", "batch_size", "num_workers", "pin_memory", "drop_last",
                                            "timeout", "sampler", "prefetch_factor", "dataloader"):
            raise RuntimeError(f"Property {name} cannot be set after initialization in a DataLoader")
        return super().__setattr__(name, value)

    @classmethod
    def is_dataloader_already_wrapped(cls, dataloader: DataLoader):
        """Returns whether the ``dataloader`` is wrapped with ``cls``. This helper method checks recursively through all
        wrappings until the underlying dataloader is reached.

        Args:
            dataloader (DataLoader): The dataloader to check

        Returns:
            bool: Whether the ``dataloader`` is wrapped recursively with ``cls``.
        """
        if isinstance(dataloader, cls):
            return True
        if not isinstance(dataloader, WrappedDataLoader):
            return False
        if not isinstance(dataloader.dataloader, WrappedDataLoader):
            return False
        return cls.is_dataloader_already_wrapped(dataloader.dataloader)


def unwrap_data_loader(dataloader: DataLoader) -> DataLoader:
    """Recursively unwraps a dataloader if it is of type :class:`WrappedDataLoader`.

    Args:
        dataloader (DataLoader): The dataloader to unwrap

    Returns:
        DataLoader: The underlying dataloader
    """
    if isinstance(dataloader, WrappedDataLoader):
        return unwrap_data_loader(dataloader.dataloader)
    return dataloader


@dataclass
class DataloaderHparams(hp.Hparams):
    """Hyperparameters to initialize a :class:`~torch.utils.data.Dataloader`.

    Parameters:
        num_workers (int, optional): Number of CPU workers to use per device to fetch data.
            Set to ``0`` to use the main training thread for dataloading.
            While zero workers can be useful for debugging, it should not be used for performance reasons.
            (default: ``8``)
        prefetch_factor (int, optional): Number of samples loaded in advance by each worker.
            For example, 2 means there will be a total of 2 * num_workers samples prefetched across all workers.
            If ``num_workers = 0``, then the ``prefetch_factor`` must be left at the default value.
            (default: ``2``)
        persistent_workers (bool): Whether to reuse dataloader workers across epochs. If ``num_workers`` is 0,
            then this field must be ``False``. (default: ``True``)
        pin_memory (bool, optional): Whether or not to copy Tensors into CUDA pinned memory before returning them.
            If ``num_workers = 0``, then the ``pin_memory`` must be ``False``. (default: ``True``)
        timeout (float): Timeout, in seconds, for collecting a batch from workers. Set to ``0`` for no timeout.
            (default: ``0``)
    """

    num_workers: int = hp.optional(textwrap.dedent("""\
        Number of CPU workers to use per device to fetch data.
        Set to ``0`` to use the main training thread for dataloading.
        While zero workers can be useful for debugging, it should not be used for performance reasons."""),
                                   default=8)
    prefetch_factor: int = hp.optional(textwrap.dedent("""\
        Number of samples loaded in advance by each worker.
        For example, 2 means there will be a total of 2 * num_workers samples prefetched across all workers.
        If ``num_workers = 0``, then the ``prefetch_factor`` must be left at the default value."""),
                                       default=2)
    persistent_workers: bool = hp.optional(textwrap.dedent("""\
         Whether to reuse dataloader workers across epochs. If ``num_workers`` is 0,
            then this field must be ``False``"""),
                                           default=True)
    pin_memory: bool = hp.optional(textwrap.dedent("""\
            Whether or not to copy Tensors into CUDA pinned memory before returning them.
            If ``num_workers = 0``, then the ``pin_memory`` must be ``False``."""),
                                   default=True)
    timeout: float = hp.optional(
        "Timeout, in seconds, for collecting a batch from workers. Set to ``0`` for no timeout.", default=0)

    def initialize_object(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        sampler: Optional[torch.utils.data.Sampler[int]],
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
    ) -> DataLoader:
        """Create a dataloader.

        Args:
            dataset (Dataset): The dataset.
            batch_size (int): The per-device batch size.
            sampler (torch.utils.data.Sampler[int] or None): The sampler to use for the dataloader.
            drop_last (bool): Whether to drop the last batch if the number of
                samples is not evenly divisible by the batch size.
            collate_fn (callable, optional): Custom collate function. Defaults to None.
            worker_init_fn (callable, optional): Custom worker init function. Defaults to None.

        Returns:
            DataLoader: The dataloader.
        """

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           drop_last=drop_last,
                                           sampler=sampler,
                                           collate_fn=collate_fn,
                                           worker_init_fn=worker_init_fn,
                                           timeout=self.timeout,
                                           prefetch_factor=self.prefetch_factor,
                                           persistent_workers=self.persistent_workers)
