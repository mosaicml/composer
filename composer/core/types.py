# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple, Union

import torch
import torch.utils.data
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric

from composer.core.algorithm import Algorithm
from composer.core.event import Event as Event
from composer.core.logging import Logger as Logger
from composer.core.precision import Precision as Precision
from composer.core.serializable import Serializable as Serializable
from composer.core.state import State as State

Tensor = torch.Tensor
Tensors = Union[Tensor, Tuple[Tensor, ...], List[Tensor]]
"""
The Mosaic Trainer supports multiple type of Batches returned
from the dataloader:
* (Tensors, Tensors): pair of tensors for the target and labels.
* Dict[str, Any]: dictionary of data, typically used in NLP.
"""
BatchPair = Sequence[Tensors]
BatchDict = Dict[str, Tensor]
Batch = Union[BatchPair, BatchDict, Tensor]


def as_batch_dict(batch: Batch) -> BatchDict:
    if not isinstance(batch, dict):
        raise TypeError(f'batch_dict requires batch of type dict, got {type(batch)}')
    return batch


def as_batch_pair(batch: Batch) -> BatchPair:
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f'batch_pair required batch to be a tuple or list, got {type(batch)}')
    if not len(batch) == 2:
        raise TypeError(f'batch has length {len(batch)}, expected length 2')
    return batch


Dataset = torch.utils.data.Dataset[Batch]


class BreakEpochException(Exception):
    pass


class DataLoader(Protocol):
    dataset: Dataset
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: torch.utils.data.Sampler[int]
    prefetch_factor: int

    def __iter__(self) -> Iterator[Batch]:
        ...

    def __len__(self) -> int:
        ...


Metrics = Union[Metric, MetricCollection]

Optimizer = torch.optim.Optimizer
Optimizers = Union[Optimizer, Tuple[Optimizer, ...]]
Scheduler = torch.optim.lr_scheduler._LRScheduler
Schedulers = Union[Scheduler, Tuple[Scheduler, ...]]

Scaler = torch.cuda.amp.grad_scaler.GradScaler

Model = torch.nn.Module
ModelParameters = Union[Iterable[Tensor], Iterable[Dict[str, Tensor]]]

Algorithms = Sequence[Algorithm]

JSON = Union[str, float, int, None, List['JSON'], Dict[str, 'JSON']]

TPrefetchFn = Callable[[Batch], Batch]

StateDict = Dict[str, Any]
