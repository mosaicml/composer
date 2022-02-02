# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed
import torch.utils.data
import yahp as hp

from composer.core.types import Batch, DataLoader, Dataset


class WrappedDataLoader(DataLoader):

    def __init__(self, dataloader: DataLoader) -> None:
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
        num_workers (int): Number of CPU workers to use per device to fetch data.
        prefetch_factor (int): Number of samples loaded in advance by each worker.
            2 means there will be a total of 2 * num_workers samples prefetched across all workers.
        persistent_workers (bool): Whether or not to shutdown workers after the dataset has been consumed once.
        pin_memory (bool): Whether or not to copy Tensors into CUDA pinned memory before returning them.
        timeout (float): Timeout, in seconds, for collecting a batch from workers. Set to 0 for no timeout.
    
    """

    num_workers: int = hp.required("Number of CPU workers to use per device to fetch data.", template_default=8)
    prefetch_factor: int = hp.required("Number of samples loaded in advance by each worker", template_default=2)
    persistent_workers: bool = hp.required("Whether to shutdown workers after the dataset has been consumed once",
                                           template_default=True)
    pin_memory: bool = hp.required("Whether to copy Tensors into CUDA pinned memory before returning them",
                                   template_default=True)
    timeout: float = hp.required("Timeout, in seconds, for collecting a batch from workers. Set to 0 for no timeout",
                                 template_default=0)

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
