# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Specifications for operating and training on data."""
from __future__ import annotations

import collections.abc
import logging
import textwrap
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

from composer.utils import dist, ensure_tuple

if TYPE_CHECKING:
    from composer.core.types import Batch

__all__ = ['DataSpec', 'ensure_data_spec']

log = logging.getLogger(__name__)


def _split_list(l, microbatch_size: int):
    if len(l) < microbatch_size:
        warnings.warn(
            f'Cannot split list of length {len(l)} into batches of size {microbatch_size}. '
            'As it is smaller, no splitting will be done. This may happen on the last batch '
            'of a dataset if it is a smaller size than the microbatch size.',
        )
        microbatch_size = len(l)
    return [l[start:start + microbatch_size] for start in range(0, len(l), microbatch_size)]


def _split_tensor(t, microbatch_size: int):
    if len(t) < microbatch_size:
        warnings.warn(
            f'Cannot split tensor of length {len(t)} into batches of size {microbatch_size}. '
            'As it is smaller, no splitting will be done. This may happen on the last batch '
            'of a dataset if it is a smaller size than the microbatch size.',
        )
        microbatch_size = len(t)
    return t.split(microbatch_size)


def _split_mapping(m, microbatch_size: int):
    chunked = {}
    for k, v in m.items():
        if isinstance(v, torch.Tensor):
            chunked[k] = _split_tensor(v, microbatch_size)
        elif isinstance(v, (list, tuple)):
            chunked[k] = _split_list(v, microbatch_size)
        elif isinstance(v, Mapping):
            chunked[k] = _split_mapping(v, microbatch_size)
        elif isinstance(v, (int, float, str, bool)):
            # Defer broadcasting primitives until we know num_chunks
            pass
        else:
            raise ValueError(f'Unsupported batch type: {type(v)}.')
    num_chunks = 1  # Default to 1 chunks if there are no tensors or everything is primitive
    if len(chunked.keys()) != 0:
        num_chunks = len(list(chunked.values())[0])
    # Broadcast primitives to all chunks
    for k, v in m.items():
        if isinstance(v, (int, float, str, bool)):
            chunked[k] = [v] * num_chunks
    return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]


def _check_list_is_primitives(l):
    """Checks if all elements in a list are the same primitive type."""
    if len(l) == 0:
        return True
    first_type = type(l[0])
    if not isinstance(l[0], (int, float, str, bool)):
        return False
    for item in l:
        if type(item) != first_type:
            return False
    return True


def _default_split_batch(batch: Any, microbatch_size: Union[int, float]) -> Sequence:
    """Splits batch into chunks of size `microbatch_size` for gradient accumulation.

    Works with tensors, dictionaries of tensors, (x, y) tuples, and lists where ``batch`` is the 2nd dimension.

    Args:
        batch (Any): output from the dataloader.
        microbatch_size (int | float): Size of microbatches to batch into.
    """
    if isinstance(microbatch_size, float):
        raise ValueError('_default_split_batch does not support floating point microbatch_size.')

    if isinstance(batch, torch.Tensor):  # check for a single stack of tensors
        return _split_tensor(batch, microbatch_size)
    elif isinstance(batch, Mapping):  # check for dictionary (hf style)
        return _split_mapping(batch, microbatch_size)
    elif isinstance(batch, (tuple, list)) and _check_list_is_primitives(batch):  # check for list of primitives
        return _split_list(batch, microbatch_size)
    elif isinstance(batch, (tuple, list)):  # check for batch on 2nd dimension
        result = []
        for item in batch:
            if isinstance(item, torch.Tensor):
                result.append(_split_tensor(item, microbatch_size))
            elif isinstance(item, (list, tuple)):
                result.append(_split_list(item, microbatch_size))
            else:
                raise ValueError(f'Unsupported batch type: {type(item)}.')
        return list(zip(*result))
    raise NotImplementedError(
        textwrap.dedent(
            """\
            The default `split_fn` is unable to split the output of this dataloader. To enable microbatching,
             please and specify a `DataSpec` with `split_batch` for your dataset.""",
        ),
    )


default_split_batch = _default_split_batch


class DataSpec:
    """Specifications for operating and training on data.

    An example of constructing a :class:`DataSpec` object with a ``batch_transforms``
    callable and then using it with :class:`~.Trainer`:

    .. doctest::

       >>> # Construct DataSpec and subtract mean from the batch
       >>> batch_transform_fn = lambda xs, ys: (xs.sub_(xs.mean()), ys)
       >>> train_dspec = DataSpec(train_dataloader, batch_transforms=batch_transform_fn)
       >>> # The same function can be used for eval dataloader as well
       >>> eval_dspec = DataSpec(eval_dataloader, batch_transforms=batch_transform_fn)
       >>> # Use this DataSpec object to construct trainer
       >>> trainer = Trainer(
       ...     model=model,
       ...     train_dataloader=train_dspec,
       ...     eval_dataloader=eval_dspec,
       ...     optimizers=optimizer,
       ...     max_duration="1ep",
       ... )

    Args:
        dataloader (Union[Iterable, torch.utils.data.DataLoader]): The dataloader, which can be any iterable that yields batches.

        num_samples (int, optional): The total number of samples in an epoch, across all ranks. This field is used by
            the :class:`.Timestamp` (training progress tracker). If not specified, then ``len(dataloader.dataset)`` is
            used (if this property is available). Otherwise, the dataset is assumed to be unsized.

        num_tokens (int, optional): The total number of tokens in an epoch. This field is used by the
            :class:`.Timestamp` (training progress tracker).

        batch_transforms ((Batch) -> Batch, optional): Function called by the :class:`.Trainer` to modify the
            batch before it is moved onto the device. For example, this function can be used for CPU-based
            normalization. It can modify the batch in-place, and it should return the modified batch. If not specified,
            the batch is not modified.

        microbatch_transforms ((Batch) -> Batch, optional): Function called by the :class:`.Trainer` to modify the
            microbatch before it is moved onto the device. For example, this function can be used for GPU-based
            normalization. It can modify the microbatch in-place, and it should return the modified microbatch. If not
            specified, the microbatch is not modified.

        split_batch ((Batch, (int | float)) -> Sequence[Batch], optional): Function called by the :class:`.Trainer` to
            split a batch (the first parameter) into microbatches of a given size (the second parameter). If
            the ``dataloader`` yields batches not of type :class:`torch.Tensor`, Mapping, tuple, or list, then
            this function must be specified.

        get_num_samples_in_batch ((Batch) -> Union[int, float], optional): Function that is called by the :class:`.Trainer`
            to get the number of samples in the provided batch.

            By default, if the batch contains tensors that all have the same 0th dim, then the value of the 0th dim will
            be returned. If the batch contains tensors where the 0th dim differ, then this function must be specified.

        get_num_tokens_in_batch ((Batch) -> int, optional): Function that is called by the :class:`.Trainer` to
            get the number of tokens in the provided batch.

            By default, it checks for HuggingFace-style dictionary batches with ``input_ids``, and then checks ``dataset.max_seq_len``, and returns 0
            if both of those fail, meaning that number of tokens processed will not be tracked as a part of the training progress tracking.
            Note that the defaults do NOT take padding into account, so if you want the token calculation to exclude padding, you should specify this function.
            This function must be specified to track the number of tokens processed during training in a non-default way.
    """

    def __init__(
        self,
        dataloader: Union[Iterable, torch.utils.data.DataLoader],
        num_samples: Optional[int] = None,
        num_tokens: Optional[int] = None,
        batch_transforms: Optional[Callable[[Batch], Batch]] = None,
        microbatch_transforms: Optional[Callable[[Batch], Batch]] = None,
        split_batch: Optional[Callable[[Batch, Union[int, float]], Sequence[Batch]]] = None,
        get_num_samples_in_batch: Optional[Callable[[Batch], Union[int, float]]] = None,
        get_num_tokens_in_batch: Optional[Callable[[Batch], Union[int, dict[str, int]]]] = None,
    ) -> None:
        self.dataloader: Union[Iterable, torch.utils.data.DataLoader] = dataloader
        self.num_tokens = num_tokens
        self.batch_transforms = self._default_transforms if batch_transforms is None else batch_transforms
        self.microbatch_transforms = self._default_transforms if microbatch_transforms is None else microbatch_transforms
        self.split_batch = default_split_batch if split_batch is None else split_batch
        self.get_num_samples_in_batch = self._default_get_num_samples_in_batch if get_num_samples_in_batch is None else get_num_samples_in_batch
        self._get_num_tokens_in_batch = self._default_get_num_tokens_in_batch if get_num_tokens_in_batch is None else get_num_tokens_in_batch

        if num_samples is not None:
            self.num_samples = num_samples
        else:
            if isinstance(
                dataloader,
                torch.utils.data.DataLoader,
            ) and isinstance(dataloader.dataset, collections.abc.Sized):
                try:
                    self.num_samples = len(dataloader.dataset)
                except (TypeError, NotImplementedError):
                    self.num_samples = None
            else:
                self.num_samples = None

        if isinstance(dataloader, torch.utils.data.DataLoader):
            if dataloader._iterator is not None:
                raise ValueError((
                    'The dataloader has an active iterator. This could occur '
                    'if `persistent_workers=True` and the dataloader has already been iterated, '
                    'or if the dataloader is mid-epoch. It is required that the training dataloader '
                    'does not have an active iterator, so CPU dataset augmentations can be '
                    'correctly inserted. To fix, please do not iterate over the dataloader before passing it into '
                    'the Trainer.'
                ))
            world_size = dist.get_world_size()
            # Check for Distributed Sampler if not using IterableDataset on more than 1 GPU
            if world_size > 1 and not isinstance(dataloader.dataset, torch.utils.data.IterableDataset):
                is_sampler_distributed = isinstance(dataloader.sampler, DistributedSampler)
                is_batch_sampler_distributed = dataloader.batch_sampler is not None and isinstance(
                    dataloader.batch_sampler,
                    DistributedSampler,
                )
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
                        'For more information, see https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler.',
                    )

    def _default_transforms(self, batch: Batch):
        return batch

    def _default_get_num_samples_in_batch(self, batch: Batch) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]

        dim0_sizes = []
        if isinstance(batch, (list, tuple)):
            for tensors in batch:
                for t in ensure_tuple(tensors):
                    if not hasattr(t, 'shape'):
                        raise ValueError(
                            'Unable to determine the batch size, batch contains'
                            f'an element of type {type(t)}, which does not have a'
                            'shape. Please use a DataSpec and provide a'
                            '`get_num_samples_in_batch(your_batch) -> int` method.',
                        )
                    dim0_sizes.append(t.shape[0])
        elif isinstance(batch, Mapping):
            for t in batch.values():
                if isinstance(t, torch.Tensor):
                    dim0_sizes.append(t.shape[0])
                elif isinstance(t, list):
                    dim0_sizes.append(len(t))
                else:
                    raise ValueError(
                        'Unable to determine the batch size as batch is a dict '
                        f'with an element of type {type(t)} which is not Tensor '
                        'or list. Please use a DataSpec and provide a '
                        '`get_num_samples_in_batch(your_batch) -> int` method.',
                    )

        if len(set(dim0_sizes)) == 1:
            return dim0_sizes[0]
        else:
            raise NotImplementedError(
                textwrap.dedent(
                    f"""\
                    Cannot determine the batch size, as multiple Tensors of
                    different lengths were found in the batch: sizes in batch: {dim0_sizes}.
                    Please use a DataSpec and specify `get_num_samples_in_batch`.""",
                ),
            )

    def _default_get_num_tokens_in_batch(self, batch: Batch) -> int:
        # First try HuggingFace-style input dicts
        if isinstance(batch, Mapping) and 'input_ids' in batch:
            samples_per_batch = batch['input_ids'].shape[0]
            return batch['input_ids'].shape[1] * samples_per_batch
        # Then try dataset.max_seq_len
        elif hasattr(self.dataloader, 'dataset') and hasattr(self.dataloader.dataset, 'max_seq_len'):  # type: ignore
            samples_per_batch = self.get_num_samples_in_batch(batch)
            return self.dataloader.dataset.max_seq_len * samples_per_batch  # type: ignore
        return 0

    def get_num_tokens_in_batch(self, batch: Batch, token_type: str = 'total') -> int:
        num_tokens = self._get_num_tokens_in_batch(batch)

        if isinstance(num_tokens, int):
            return num_tokens
        elif isinstance(num_tokens, dict):
            if token_type not in num_tokens:
                raise ValueError(f'Token type {token_type} not found in num_tokens dict.')
            return num_tokens[token_type]
        else:
            raise ValueError(f'Unexpected return type from get_num_tokens_in_batch: {type(num_tokens)}')


def ensure_data_spec(dataloader: Union[DataSpec, Iterable, dict]) -> DataSpec:
    """Ensures that the ``dataloader`` is a :class:`.DataSpec`.

    Args:
        dataloader (DataSpec | Iterable | dict): A DataSpec, DataLoader, or dict of DataSpec kwargs.

    Returns:
        DataSpec: A DataSpec
    """
    if isinstance(dataloader, dict):
        # treat as kwargs for DataSpec
        dataloader = DataSpec(**dataloader)
    if not isinstance(dataloader, DataSpec):
        dataloader = DataSpec(dataloader)

    return dataloader
