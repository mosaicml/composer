# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, cast

import torch
import torch.distributed
import torch.utils.data
import yahp as hp
from torch.utils.data.distributed import DistributedSampler

from composer.core.types import Batch, DataLoader, Dataset, SamplerFactory
from composer.utils import ddp


class WrappedDataLoader(DataLoader):

    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.dataloader)

    def __bool__(self) -> bool:
        return True

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'dataloader' and not hasattr(self, 'dataloader'):
            return super().__setattr__(name, value)
        if name in ("dataset", "batch_sampler", "batch_size", "num_workers", "pin_memory", "drop_last", "timeout",
                    "sampler", "prefetch_factor", "dataloader"):
            raise RuntimeError(f"Property {name} cannot be set after initialization in a DataLoader")
        return setattr(self.dataloader, name, value)

    def __getattribute__(self, name: str) -> Any:
        if name == 'dataloader':
            return super().__getattribute__(name)
        return getattr(self.dataloader, name)


class DDPDataLoader(WrappedDataLoader):
    """Wraps the dataset to ensure that, if the dataset sampler is a
    :class:`~torch.utils.data.distributed.DistributedSampler`, then
    :meth:`~torch.utils.data.distributed.DistributedSampler.set_epoch`
    is called after each epoch.

    If the dataset sampler is not a :class:`~torch.utils.data.distributed.DistributedSampler`,
    then this wrapper is a no-op.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__(dataloader)
        self._iterator: Optional[Iterator[Batch]] = None

    def __iter__(self) -> DDPDataLoader:
        if self._iterator is not None:
            warnings.warn(
                "DataloaderMultipleIterationWarning: "
                "The dataloader detected the start of a new iteration before the previous iteration finished. "
                "The dataloader is skipping ahead to the start of the next epoch. "
                "Multiple simultaneous iterations through the DDP dataloader prohibited, since "
                "it automatically tracks the current epoch.")
            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(epoch=self.sampler.epoch + 1)
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self) -> Batch:
        assert self._iterator is not None
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = None
            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(epoch=self.sampler.epoch + 1)
            raise


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
    persistent_workers: bool = hp.required(textwrap.dedent("""Whether or not to shutdown workers after the dataset
        has been consumed once"""),
                                           template_default=True)
    pin_memory: bool = hp.required(textwrap.dedent("""Whether or not to copy Tensors into CUDA pinned memory
        before returning them"""),
                                   template_default=True)
    timeout: float = hp.required("Timeout, in seconds, for collecting a batch from workers. Set to 0 for no timeout",
                                 template_default=0)

    # XXX should actually be of type SamplerFactory; or we just make this
    # not take in the factory as a field...maybe a dict with known keys?
    batch_sampler_factory: Optional[str] = hp.optional(
        "Function returning an alternate batch_sampler to use, rather than the torch DataLoader default.", default=None)

    def initialize_object(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        drop_last: bool,
        split: str = 'train',
        shuffle: bool = True,
        sampler: Optional[torch.utils.data.Sampler] = None,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
    ) -> DataLoader:
        """Create a dataloader.

        Args:
            dataset (Dataset): The dataset.
            batch_size (int): The per-device batch size.
            drop_last (bool): Whether to drop the last batch if the number of
                samples is not evenly divisible by the batch size.
            shuffle (bool): Whether to shuffle the dataset.
            collate_fn (callable, optional): Custom collate function. Defaults to None.
            worker_init_fn (callable, optional): Custom worker init function. Defaults to None.

        Returns:
            DataLoader: The dataloader.
        """
        if self.batch_sampler_factory is not None:
            if sampler is not None:
                raise RuntimeError("Can't specify both sampler and batch_sampler!")
            self.batch_sampler_factory = cast(SamplerFactory, self.batch_sampler_factory)
            batch_sampler = ddp.get_sampler(
                dataset,
                drop_last=drop_last,
                shuffle=shuffle,
                batch_size=batch_size,
                split=split,
                factory=self.batch_sampler_factory,
            )
            sampler_dependent_kwargs = dict(batch_sampler=batch_sampler)

            # total hack to avoid logging error from yaml being unable to
            # dump objects:
            # File ".../composer/composer/loggers/file_logger.py", line 99, in init
            #     yaml.safe_dump(self.config, stream=self.file)
            self.batch_sampler_factory = None

        else:
            if sampler is None:
                sampler = ddp.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)
            sampler_dependent_kwargs = dict(batch_size=batch_size, drop_last=drop_last, sampler=sampler)

        return torch.utils.data.DataLoader(dataset,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           collate_fn=collate_fn,
                                           worker_init_fn=worker_init_fn,
                                           timeout=self.timeout,
                                           prefetch_factor=self.prefetch_factor,
                                           persistent_workers=self.persistent_workers,
                                           **sampler_dependent_kwargs)
