# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import collections
import textwrap
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Sequence

import torch
import torch.distributed
import torch.utils.data
import yahp as hp

from composer.core.types import Batch, DataLoader, Dataset
from composer.utils.iter_helpers import ensure_tuple


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


class ComposerDataLoader(WrappedDataLoader):
    """Wrapper for dataloaders to hold additional data-related metadata.

    Args:
        dataloader (DataLoader): The dataloader.

        num_samples (int, optional). If specified, the total number of samples in the dataset, across all ranks.
            If not specified, then ``len(dataloader.dataset)`` is used (if this property is available).
            Otherwise, the dataset is assumed to be unsized.

        num_tokens (int, optional): If specified, the total number of tokens in the dataset.

        device_transforms ((Batch) -> Batch, optional): Function that is called by the trainer to modify the batch
            once it has been moved onto the device. For example, this function can be used for GPU-based normalization.
            It can modify the batch in-place, and it should return the modified batch. If omitted, the batch is not
            modified.

        split_batch ((Batch, int) -> Sequence[Batch], optional): Function that is called by the trainer to split a
            batch (the first parameter) into the number of microbatches specified by the second parameter.
            By default, batches of type :class:`BatchPair` can be split automatically. If the
            :attr:`dataloader` yields batches of a different type, then this function must be specified.

        get_num_samples_in_batch ((Batch) -> int, optional): Function that is called by the trainer to
            get the number of samples in the provided batch.

            By default, if the batch contains tensors that all have the same length, then that
            length will be returned. If the batch contains tensors where the lengths differ,
            then this function must be specified.

        get_num_tokens_in_batch ((Batch) -> int, optional): Function that is called by the trainer to
            get the number of tokens in the provided batch.

            By default, it returns 0, meaning that tokens will not be tracked.
            This function must be specified to track tokens.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
        num_tokens: Optional[int] = None,
        device_transforms: Optional[Callable[[Batch], Batch]] = None,
        split_batch: Optional[Callable[[Batch, int], Sequence[Batch]]] = None,
        get_num_samples_in_batch: Optional[Callable[[Batch], int]] = None,
        get_num_tokens_in_batch: Optional[Callable[[Batch], int]] = None,
    ) -> None:
        super().__init__(dataloader)
        self.num_tokens = num_tokens
        self.device_transforms = self._default_device_transforms if device_transforms is None else device_transforms
        self.split_batch = self._default_split_batch if split_batch is None else split_batch
        self.get_num_samples_in_batch = self._default_get_num_samples_in_batch if get_num_samples_in_batch is None else get_num_samples_in_batch
        self.get_num_tokens_in_batch = self._default_get_num_tokens_in_batch if get_num_tokens_in_batch is None else get_num_tokens_in_batch
        if num_samples is not None:
            self.num_samples = num_samples

        else:
            if isinstance(dataloader.dataset, collections.abc.Sized):
                try:
                    self.num_samples = len(dataloader.dataset)
                except (TypeError, NotImplementedError):
                    self.num_samples = None
            else:
                self.num_samples = None

    def _default_device_transforms(self, batch: Batch):
        return batch

    def _default_split_batch(self, batch: Batch, num_microbatches: int) -> Sequence[Batch]:
        if not isinstance(batch, Sequence):
            raise ValueError(f'split_fn requires batch be a tuple pair of tensors, got {type(batch)}')
        x, y = batch
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return list(zip(x.chunk(num_microbatches), y.chunk(num_microbatches)))
        if isinstance(x, List) and isinstance(y, List):
            return list(
                zip(
                    [x[i::num_microbatches] for i in range(num_microbatches)],
                    [y[i::num_microbatches] for i in range(num_microbatches)],
                ))
        raise NotImplementedError(
            textwrap.dedent("""\
                The default split_fn is unable to split the output of this
                dataloader. Please wrap you dataloader with `ComposerDataLoader` 
                and specify `split_batch`."""))

    def _default_get_num_samples_in_batch(self, batch: Batch) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]

        dim0_sizes = []
        if isinstance(batch, (list, tuple)):
            for tensors in batch:
                for t in ensure_tuple(tensors):
                    dim0_sizes.append(t.shape[0])
        elif isinstance(batch, dict):
            dim0_sizes = [t.shape[0] for t in batch.values()]

        if len(set(dim0_sizes)) == 1:
            return dim0_sizes[0]
        else:
            raise NotImplementedError(
                textwrap.dedent(f"""\
                    Cannot determine the batch size, as multiple Tensors of
                    different lengths were found in the batch: sizes in batch: {dim0_sizes}.
                    Please wrap you dataloader with `ComposerDataLoader` and specify 
                    `get_num_samples_in_batch`."""))

    def _default_get_num_tokens_in_batch(self, batch: Batch) -> int:
        del batch  # unused
        return 0


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
