# Copyright 2021 MosaicML. All Rights Reserved.

"""Reference for common types used throughout our library.

See :doc:`/core/types` for documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.utils.data
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric

from composer.core.algorithm import Algorithm as Algorithm
from composer.core.data_spec import DataSpec as DataSpec
from composer.core.event import Event as Event
from composer.core.logging import Logger as Logger
from composer.core.precision import Precision as Precision
from composer.core.serializable import Serializable as Serializable
from composer.core.state import State as State
from composer.core.time import Time as Time
from composer.core.time import Timer as Timer
from composer.core.time import TimeUnit as TimeUnit
from composer.utils.string_enum import StringEnum

try:
    from typing import Protocol
except ImportError:
    Protocol = object  # Protocol is not available in python 3.7

if TYPE_CHECKING:
    from typing import Protocol

Tensor = torch.Tensor
Tensors = Union[Tensor, Tuple[Tensor, ...], List[Tensor]]

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
        batch_size (int, optional): How many samples per batch to load
            (default: ``1``).
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


@dataclass
class Evaluator:
    """Wrapper for a dataloader to include metrics that apply to a specific
    dataset.

    Attributes:
        label (str): Name of the Evaluator
        dataloader (Union[DataLoader, DataSpec]): Dataloader/DataSpec for evaluation data
        metrics (Optional[Metrics]): Metrics to use for the dataset.
        validate_every_n_batches (Optional[int]): Compute metrics on evaluation data every N batches.
             Set to -1 to never validate on a batchwise frequency. (default: ``-1``)
        validate_every_n_epochs (Optional[int]): Compute metrics on evaluation data every N epochs.
            Set to -1 to never validate on a epochwise frequency. (default: ``1``)
        metric_names: (Optional[List[str]]): List of string names for desired metrics in an Evaluator. If specified,
            the trainer will look through the compatible metrics for a model and populate the metrics field
            with torchmetrics with names appearing in metric_names.
        eval_subset_num_batches (int, optional): If specified, evaluate on this many batches.
            This parameter has no effect if it is greater than ``len(eval_dataloader)``.
            If None (the default), then the entire dataloader will be iterated over.
        device_transforms ((Batch) -> Batch, optional): Function that is called by the trainer to modify the batch
            once it has been moved onto the device. For example, this function can be used for GPU-based normalization.
            It can modify the batch in-place, and it should return the modified batch. If omitted, the batch is not
            modified.
    """

    label: str
    dataloader: Union[DataLoader, DataSpec]
    metrics: Metrics = None
    validate_every_n_epochs: int = 1
    validate_every_n_batches: int = -1
    metric_names: Optional[Sequence[str]] = None
    eval_subset_num_batches: Optional[int] = None
    device_transforms: Optional[Callable[[Batch], Batch]] = None
    _data_spec: Optional[DataSpec] = None

    def __post_init__(self):
        if isinstance(self.dataloader, DataSpec):
            dataloader_spec = self.dataloader
            self.device_transforms = dataloader_spec.device_transforms
        else:
            dataloader_spec = DataSpec(self.dataloader)
        self.dataloader = dataloader_spec.dataloader
        self._data_spec = dataloader_spec

        if self.metrics is not None:
            assert isinstance(self.metrics, (Metric, MetricCollection)), \
            "   Error module.metrics() must return a Metric or MetricCollection object."
            if isinstance(self.metrics, Metric):
                # Forcing metrics to be a MetricCollection simplifies logging results
                self.metrics = MetricCollection([self.metrics])



Metrics = Union[Metric, MetricCollection]

Optimizer = torch.optim.Optimizer
Optimizers = Union[Optimizer, Tuple[Optimizer, ...], List[Optimizer]]
Scheduler = torch.optim.lr_scheduler._LRScheduler
Schedulers = Union[Scheduler, Tuple[Scheduler, ...], List[Scheduler]]

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
