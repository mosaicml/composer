# Copyright 2021 MosaicML. All Rights Reserved.
from __future__ import annotations

import collections.abc
import textwrap
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

import torch

from composer.utils.iter_helpers import ensure_tuple

if TYPE_CHECKING:
    from composer.core.types import Batch, DataLoader


class DataSpec:
    """Specification for describing how to train and operate on data.

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
        self.dataloader = dataloader
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
                dataloader. Please use a DataSpec and specify `split_batch`."""))

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
                textwrap.dedent(f"""
                    Cannot determine the batch size, as multiple Tensors of
                    different lengths were found in the batch: sizes in batch: {dim0_sizes}.
                    Please use a DataSpec and specify `get_num_samples_in_batch`."""))

    def _default_get_num_tokens_in_batch(self, batch: Batch) -> int:
        del batch  # unused
        return 0
