# Copyright 2021 MosaicML. All Rights Reserved.

"""Reference for common types used throughout our library.

TODO: This attributes list is incomplete.

Attributes:
    Model (torch.nn.Module): Alias for :class:`torch.nn.Module`.
    ModelParameters (Iterable[Tensor] | Iterable[Dict[str, Tensor]]): Type alias for model parameters used to
        initialize optimizers.
    Tensors (Tensor | Tuple[Tensor, ...] | List[Tensor]): Commonly used to represent e.g. a set of inputs,
        where it is unclear whether each input has its own tensor, or if all the inputs are concatenated in a single
        tensor.
    Batch (BatchPair | BatchDict | Tensor): Union type covering the most common representations of batches.
        A batch of data can be represented in several formats, depending on the application.
    BatchPair (Tuple[Tensors, Tensors] | List[Tensor]): Commonly used in computer vision tasks. The object is assumed
        to contain exactly two elements, where the first represents inputs and the second represents targets.
    BatchDict (Dict[str, Tensor]): Commonly used in natural language processing tasks.
    Metrics (Metric | MetricCollection): Union type covering common formats for representing metrics.
    Optimizer (torch.optim.Optimizer): Alias for :class:`torch.optim.Optimizer`
    Optimizers (Optimizer | List[Optimizer] | Tuple[Optimizer, ...]): Union type for indeterminate amounts of optimizers.
    Scheduler (torch.optim.Optimizer): Alias for :class:`torch.optim.lr_scheduler._LRScheduler`
    Schedulers (Scheduler | List[Scheduler] | Tuple[Scheduler, ...]): Union type for indeterminate amounts of schedulers.
    Scaler (torch.cuda.amp.grad_scaler.GradScaler): Alias for :class:`torch.cuda.amp.grad_scaler.GradScaler`.
    JSON (str | float | int | None | List['JSON'] | Dict[str, 'JSON']): JSON Data
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union

import torch
import torch.utils.data
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric

from composer.core.algorithm import Algorithm as Algorithm
from composer.core.data_spec import DataSpec as DataSpec
from composer.core.evaluator import Evaluator as Evaluator
from composer.core.event import Event as Event
from composer.core.logging import Logger as Logger
from composer.core.precision import Precision as Precision
from composer.core.serializable import Serializable as Serializable
from composer.core.state import State as State
from composer.utils.string_enum import StringEnum

try:
    from typing import Protocol
except ImportError:
    Protocol = object  # Protocol is not available in python 3.7

if TYPE_CHECKING:
    from typing import Protocol

T = TypeVar('T')
Many = Union[T, Tuple[T, ...], List[T]]

Tensor = torch.Tensor
Tensors = Many[Tensor]

# For BatchPar, if it is a list, then it should always be of length 2.
# Pytorch's default collate_fn returns a list even when the dataset returns a tuple.
BatchPair = Union[Tuple[Tensors, Tensors], List[Tensor]]
BatchDict = Dict[str, Tensor]
Batch = Union[BatchPair, BatchDict, Tensor]


def as_batch_dict(batch: Batch) -> BatchDict:
    """Casts a :class:`Batch` as a :class:`BatchDict`.

    Args:
        batch (Batch): A batch.
    Raises:
        TypeError: If the batch is not a :class:`BatchDict`.
    Returns:
        BatchDict: The batch, represented as a :class:`BatchDict`.
    """

    if not isinstance(batch, dict):
        raise TypeError(f'batch_dict requires batch of type dict, got {type(batch)}')
    return batch


def as_batch_pair(batch: Batch) -> BatchPair:
    """Casts a :class:`Batch` as a :class:`BatchPair`.

    Args:
        batch (Batch): A batch.
    Returns:
        BatchPair: The batch, represented as a :class:`BatchPair`.
    Raises:
        TypeError: If the batch is not a :class:`BatchPair`.
    """

    if not isinstance(batch, (tuple, list)):
        raise TypeError(f'batch_pair required batch to be a tuple or list, got {type(batch)}')
    if not len(batch) == 2:
        raise TypeError(f'batch has length {len(batch)}, expected length 2')
    return batch


Dataset = torch.utils.data.Dataset[Batch]


class BreakEpochException(Exception):
    """Raising this exception will immediately end the current epoch.

    If you're wondering whether you should use this, the answer is no.
    """

    pass


class DataLoader(Protocol):
    """Protocol for custom DataLoaders compatible with
    :class:`torch.utils.data.DataLoader`.

    Attributes:
        dataset (Dataset): Dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load for a
            single device (default: ``1``).
        num_workers (int): How many subprocesses to use for data loading.
            ``0`` means that the data will be loaded in the main process.
        pin_memory (bool): If ``True``, the data loader will copy Tensors
            into CUDA pinned memory before returning them.
        drop_last (bool): If ``len(dataset)`` is not evenly
            divisible by :attr:`batch_size`, whether the last batch is
            dropped (if True) or truncated (if False).
        timeout (float): The timeout for collecting a batch from workers.
        sampler (torch.utils.data.Sampler[int]): The dataloader sampler.
        prefetch_factor (int): Number of samples loaded in advance by each
            worker. ``2`` means there will be a total of
            2 * :attr:`num_workers` samples prefetched across all workers.
    """

    dataset: Dataset
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: torch.utils.data.Sampler[int]
    prefetch_factor: int

    def __iter__(self) -> Iterator[Batch]:
        """Iterates over the dataset.

        Yields:
            Iterator[Batch]: An iterator over batches.
        """
        ...

    def __len__(self) -> int:
        """Returns the number of batches in an epoch.

        Raises:
            NotImplementedError: Raised if the dataset has unknown length.

        Returns:
            int: Number of batches in an epoch.
        """
        ...


Evaluators = Many[Evaluator]
Metrics = Union[Metric, MetricCollection]
Optimizer = torch.optim.Optimizer
Optimizers = Many[Optimizer]
Scheduler = torch.optim.lr_scheduler._LRScheduler
Schedulers = Many[Scheduler]

Scaler = torch.cuda.amp.grad_scaler.GradScaler

Model = torch.nn.Module
ModelParameters = Union[Iterable[Tensor], Iterable[Dict[str, Tensor]]]

JSON = Union[str, float, int, None, List['JSON'], Dict[str, 'JSON']]

StateDict = Dict[str, Any]


class MemoryFormat(StringEnum):
    CONTIGUOUS_FORMAT = "contiguous_format"
    CHANNELS_LAST = "channels_last"
    CHANNELS_LAST_3D = "channels_last_3d"
    PRESERVE_FORMAT = "preserve_format"
