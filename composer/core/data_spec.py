# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Specifications for operating and training on data."""
from __future__ import annotations

import collections.abc
import math
import textwrap
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

from composer.utils import dist, ensure_tuple

if TYPE_CHECKING:
    from composer.core.types import Batch

__all__ = ['DataSpec', 'ensure_data_spec']


def _num_microbatches_split_list(l, num_microbatches: int):
    if len(l) < num_microbatches:
        raise ValueError(
            textwrap.dedent(f"""\
        Cannot split list of length {len(l)} into {num_microbatches} batches.
         make sure `grad_accum` is less than or equal to `train_batch_size // world_size`."""))

    # Note: this is to match the behavior of tensor.chunk, which is used in _split_tensor
    chunked_microbatch_size = math.ceil(len(l) / num_microbatches)
    return [l[start:start + chunked_microbatch_size] for start in range(0, len(l), chunked_microbatch_size)]


def _num_microbatches_split_tensor(t, num_microbatches: int):
    if len(t) < num_microbatches:
        raise ValueError(
            textwrap.dedent(f"""\
        Cannot split tensor of length {len(t)} into {num_microbatches} batches.
         make sure `grad_accum` is less than or equal to `train_batch_size // world_size`."""))
    return t.chunk(num_microbatches)


def _num_microbatches_split_mapping(m, num_microbatches: int):
    chunked = {}
    for k, v in m.items():
        if isinstance(v, torch.Tensor):
            chunked[k] = _num_microbatches_split_tensor(v, num_microbatches)
        if isinstance(v, (List, Tuple)):
            chunked[k] = _num_microbatches_split_list(v, num_microbatches)
    num_chunks = len(list(chunked.values())[0])
    return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]


def _num_microbatches_split_batch(batch: Any, num_microbatches: int) -> Sequence:
    """Splits batch into `num_microbatches` chunks for gradient accumulation.

    Works with tensors, dictionaries of tensors, (x, y) tuples, and lists where ``batch`` is the 2nd dimension.

    Args:
        batch (Any): output from the dataloader.
        num_microbatches (int): number of microbatches to batch into. Will be set by `grad_accum`.
    """
    if num_microbatches < 1:
        raise ValueError('num_microbatches must be at least 1')
    if num_microbatches == 1:
        return [batch]

    if isinstance(batch, torch.Tensor):  # check for a single stack of tensors
        return _num_microbatches_split_tensor(batch, num_microbatches)

    if isinstance(batch, Mapping):  # check for dictionary (hf style)
        return _num_microbatches_split_mapping(batch, num_microbatches)

    if isinstance(batch, (Tuple, List)):  # check for batch on 2nd dimension
        result = []
        for item in batch:
            if isinstance(item, torch.Tensor):
                result.append(_num_microbatches_split_tensor(item, num_microbatches))
            elif isinstance(item, (List, Tuple)):
                result.append(_num_microbatches_split_list(item, num_microbatches))
            else:
                raise ValueError(f'Unsupported batch type: {type(item)}.')
        return list(zip(*result))

    raise NotImplementedError(
        textwrap.dedent("""\
            The default `split_fn` is unable to split the output of this dataloader. To enable microbatching,
             please and specify a `DataSpec` with `split_batch` for your dataset and use `device_train_microbatch_size`."""
                       ))


def _split_list(l, microbatch_size: int):
    if len(l) < microbatch_size:
        warnings.warn(f'Cannot split list of length {len(l)} into batches of size {microbatch_size}. '
                      'As it is smaller, no splitting will be done.')
        microbatch_size = len(l)
    num_microbatches = math.ceil(len(l) / microbatch_size)
    # Note: this is to match the behavior of tensor.chunk, which is used in _split_tensor
    chunked_microbatch_size = math.ceil(len(l) / num_microbatches)
    return [l[start:start + chunked_microbatch_size] for start in range(0, len(l), chunked_microbatch_size)]


def _split_tensor(t, microbatch_size: int):
    if len(t) < microbatch_size:
        warnings.warn(f'Cannot split tensor of length {len(t)} into batches of size {microbatch_size}. '
                      'As it is smaller, no splitting will be done.')
        microbatch_size = len(t)
    num_microbatches = math.ceil(len(t) / microbatch_size)
    return t.chunk(num_microbatches)


def _split_mapping(m, microbatch_size: int):
    chunked = {}
    for k, v in m.items():
        if isinstance(v, torch.Tensor):
            chunked[k] = _split_tensor(v, microbatch_size)
        if isinstance(v, (List, Tuple)):
            chunked[k] = _split_list(v, microbatch_size)
    num_chunks = len(list(chunked.values())[0])
    return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]


def _default_split_batch(batch: Any, microbatch_size: int) -> Sequence:
    """Splits batch into chunks of size `microbatch_size` for gradient accumulation.

    Works with tensors, dictionaries of tensors, (x, y) tuples, and lists where ``batch`` is the 2nd dimension.

    Args:
        batch (Any): output from the dataloader.
        microbatch_size (int): Size of microbatches to batch into.
    """
    if isinstance(batch, torch.Tensor):  # check for a single stack of tensors
        return _split_tensor(batch, microbatch_size)
    elif isinstance(batch, Mapping):  # check for dictionary (hf style)
        return _split_mapping(batch, microbatch_size)
    elif isinstance(batch, (Tuple, List)):  # check for batch on 2nd dimension
        result = []
        for item in batch:
            if isinstance(item, torch.Tensor):
                result.append(_split_tensor(item, microbatch_size))
            elif isinstance(item, (List, Tuple)):
                result.append(_split_list(item, microbatch_size))
            else:
                raise ValueError(f'Unsupported batch type: {type(item)}.')
        return list(zip(*result))
    raise NotImplementedError(
        textwrap.dedent("""\
            The default `split_fn` is unable to split the output of this dataloader. To enable microbatching,
             please and specify a `DataSpec` with `split_batch` for your dataset."""))


class DataSpec:
    """Specifications for operating and training on data.

    An example of constructing a :class:`DataSpec` object with a ``device_transforms``
    callable (:class:`.NormalizationFn`) and then using it with :class:`~.Trainer`:

    .. doctest::

       >>> # In this case, we apply NormalizationFn
       >>> # Construct DataSpec as shown below to apply this transformation
       >>> from composer.datasets.utils import NormalizationFn
       >>> CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
       >>> CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)
       >>> device_transform_fn = NormalizationFn(mean=CHANNEL_MEAN, std=CHANNEL_STD)
       >>> train_dspec = DataSpec(train_dataloader, device_transforms=device_transform_fn)
       >>> # The same function can be used for eval dataloader as well
       >>> eval_dspec = DataSpec(eval_dataloader, device_transforms=device_transform_fn)
       >>> # Use this DataSpec object to construct trainer
       >>> trainer = Trainer(
       ...     model=model,
       ...     train_dataloader=train_dspec,
       ...     eval_dataloader=eval_dspec,
       ...     optimizers=optimizer,
       ...     max_duration="1ep",
       ... )

    Args:
        dataloader (Iterable): The dataloader, which can be any iterable that yields batches.

        num_samples (int, optional): The total number of samples in an epoch, across all ranks. This field is used by
            the :class:`.Timestamp` (training progress tracker). If not specified, then ``len(dataloader.dataset)`` is
            used (if this property is available). Otherwise, the dataset is assumed to be unsized.

        num_tokens (int, optional): The total number of tokens in an epoch. This field is used by the
            :class:`.Timestamp` (training progress tracker).

        device_transforms ((Batch) -> Batch, optional): Function called by the :class:`.Trainer` to modify the
            batch once it has been moved onto the device. For example, this function can be used for GPU-based
            normalization. It can modify the batch in-place, and it should return the modified batch. If not specified,
            the batch is not modified.

        split_batch ((Batch, int) -> Sequence[Batch], optional): Function called by the :class:`.Trainer` to
            split a batch (the first parameter) into microbatches of a given size (the second parameter). If
            the ``dataloader`` yields batches not of type :class:`torch.Tensor`, Mapping, Tuple, or List, then
            this function must be specified.

        get_num_samples_in_batch ((Batch) -> int, optional): Function that is called by the :class:`.Trainer`
            to get the number of samples in the provided batch.

            By default, if the batch contains tensors that all have the same 0th dim, then the value of the 0th dim will
            be returned. If the batch contains tensors where the 0th dim differ, then this function must be specified.

        get_num_tokens_in_batch ((Batch) -> int, optional): Function that is called by the :class:`.Trainer` to
            get the number of tokens in the provided batch.

            By default, it returns 0, meaning that number of tokens processed will not be tracked as a part of the
            training progress tracking. This function must be specified to track the number of tokens processed during
            training.
    """

    def __init__(
        self,
        dataloader: Iterable,
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
        self.split_batch = _default_split_batch if split_batch is None else split_batch
        self._num_microbatches_split_batch = _num_microbatches_split_batch  # For fallback using `grad_accum`
        self.get_num_samples_in_batch = self._default_get_num_samples_in_batch if get_num_samples_in_batch is None else get_num_samples_in_batch
        self.get_num_tokens_in_batch = self._default_get_num_tokens_in_batch if get_num_tokens_in_batch is None else get_num_tokens_in_batch

        if num_samples is not None:
            self.num_samples = num_samples
        else:
            if isinstance(dataloader, torch.utils.data.DataLoader) and isinstance(dataloader.dataset,
                                                                                  collections.abc.Sized):
                try:
                    self.num_samples = len(dataloader.dataset)
                except (TypeError, NotImplementedError):
                    self.num_samples = None
            else:
                self.num_samples = None

        if isinstance(dataloader, torch.utils.data.DataLoader):
            if dataloader._iterator is not None:
                raise ValueError(
                    ('The dataloader has an active iterator. This could occur '
                     'if `persistent_workers=True` and the dataloader has already been iterated, '
                     'or if the dataloader is mid-epoch. It is required that the training dataloader '
                     'does not have an active iterator, so CPU dataset augmentations can be '
                     'correctly inserted. To fix, please do not iterate over the dataloader before passing it into '
                     'the Trainer.'))
            world_size = dist.get_world_size()
            # Check for Distributed Sampler if not using IterableDataset on more than 1 GPU
            if world_size > 1 and not isinstance(dataloader.dataset, torch.utils.data.IterableDataset):
                is_sampler_distributed = dataloader.sampler is not None and isinstance(
                    dataloader.sampler, DistributedSampler)
                is_batch_sampler_distributed = dataloader.batch_sampler is not None and isinstance(
                    dataloader.batch_sampler, DistributedSampler)
                if not is_sampler_distributed and not is_batch_sampler_distributed:
                    raise ValueError(
                        f'The world_size({world_size}) > 1 but dataloader does not use '
                        'DistributedSampler. This will cause all ranks to train on the same '
                        'data, removing any benefit from multi-GPU training. To resolve this, '
                        'create a Dataloader with DistributedSampler. For example, '
                        'DataLoader(..., sampler=composer.utils.dist.get_sampler(...)).'
                        'Alternatively, the process group can be instantiated with '
                        'composer.utils.dist.instantiate_dist(...) and DistributedSampler can '
                        'directly be created with DataLoader(..., sampler=DistributedSampler(...)). '
                        'For more information, see https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler.'
                    )

    def _default_device_transforms(self, batch: Batch):
        return batch

    def _default_get_num_samples_in_batch(self, batch: Batch) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]

        dim0_sizes = []
        if isinstance(batch, (list, tuple)):
            for tensors in batch:
                for t in ensure_tuple(tensors):
                    if not hasattr(t, 'shape'):
                        raise ValueError('Unable to determine the batch size, batch contains'
                                         f'an element of type {type(t)}, which does not have a'
                                         'shape. Please use a DataSpec and provide a'
                                         '`get_num_samples_in_batch(your_batch) -> int` method.')
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
                    Please use a DataSpec and specify `get_num_samples_in_batch`."""))

    def _default_get_num_tokens_in_batch(self, batch: Batch) -> int:
        del batch  # unused
        return 0


def ensure_data_spec(dataloader: Union[DataSpec, Iterable, dict]) -> DataSpec:
    """Ensures that the ``dataloader`` is a :class:`.DataSpec`.

    Args:
        dataloader (DataSpec | Iterable | dict): A DataSpec, DataLoader, or Dict of DataSpec kwargs.

    Returns:
        DataSpec: A DataSpec
    """
    if isinstance(dataloader, dict):
        # treat as kwargs for DataSpec
        dataloader = DataSpec(**dataloader)
    if not isinstance(dataloader, DataSpec):
        dataloader = DataSpec(dataloader)

    return dataloader
