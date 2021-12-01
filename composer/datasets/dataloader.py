# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
import torch.distributed
import torch.utils.data
import yahp as hp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler

from composer.core.types import Batch, DataLoader
from composer.datasets.hparams import DataloaderSpec


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


class DDPDataLoader(WrappedDataLoader):
    """Ensure sampler.set_epoch() is called after each iteration.

    DDPDataLoader wraps a dataloader and a distributed sampler and is
    called after each iteration (epoch) through the dataset.
    See: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    """

    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__(dataloader)
        if not isinstance(self.dataloader.sampler, DistributedSampler):
            raise ValueError("When using the DDP data loader, the sampler must be a DistributedSampler")
        self._iterator: Optional[Iterator[Batch]] = None

    def __iter__(self) -> DDPDataLoader:
        if self._iterator is not None:
            warnings.warn(
                "DataloaderMultipleIterationWarning: "
                "The dataloader detected the start of a new iteration before the previous iteration finished. "
                "The dataloader is skipping ahead to the start of the next epoch. "
                "Multiple simultaneous iterations through the DDP dataloader prohibited, since "
                "it automatically tracks the current epoch.")
            assert isinstance(self.sampler, DistributedSampler)
            self.sampler.set_epoch(epoch=self.sampler.epoch + 1)
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self) -> Batch:
        assert self._iterator is not None
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = None
            assert isinstance(self.sampler, DistributedSampler)
            self.sampler.set_epoch(epoch=self.sampler.epoch + 1)
            raise


@dataclass
class DataloaderHparams(hp.Hparams):
    """Hyperparameters to initialize a ``torch.utils.data.Dataloader``."""

    num_workers: int = hp.required(doc="Number of CPU workers to use per gpu", template_default=8)
    prefetch_factor: int = hp.required(doc="Number of samples loaded in advance by each worker", template_default=2)
    persistent_workers: bool = hp.required(doc="Whether or not to shutdown workers after the dataset"
                                           " has been consumed once",
                                           template_default=True)
    pin_memory: bool = hp.required(doc="Whether or not to copy Tensors into CUDA pinned memory"
                                   " before returning them",
                                   template_default=True)
    timeout: int = hp.required(doc="Timeout value for collecting a batch from workers. 0 for no timeout.",
                               template_default=0)

    def initialize_object(
        self,
        batch_size: int,
        sampler: Sampler,
        dataloader_spec: DataloaderSpec,
    ) -> DataLoader:
        """Initializes the dataloader."""

        dataloader = torch.utils.data.DataLoader(
            dataloader_spec.dataset,
            batch_size=batch_size,
            shuffle=False,  # set in the sampler
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=dataloader_spec.drop_last,
            sampler=sampler,
            collate_fn=dataloader_spec.collate_fn,
            worker_init_fn=dataloader_spec.worker_init_fn,
            multiprocessing_context=dataloader_spec.multiprocessing_context,
            generator=dataloader_spec.generator,
            timeout=self.timeout,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers)
        return dataloader
