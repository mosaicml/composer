# Copyright 2021 MosaicML. All Rights Reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, NamedTuple, Optional, Sequence

import torch
import yahp as hp

from composer.core.types import Batch, Dataset, Tensor, TPrefetchFn


def _split_fn(batch: Batch, n_microbatches: int) -> List[Batch]:
    if not isinstance(batch, Sequence):
        raise ValueError(f'split_fn requires batch be a tuple pair of tensors, got {type(batch)}')
    x, y = batch
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return list(zip(x.chunk(n_microbatches), y.chunk(n_microbatches)))
    if isinstance(x, List) and isinstance(y, List):
        return list(
            zip(
                [x[i::n_microbatches] for i in range(n_microbatches)],
                [y[i::n_microbatches] for i in range(n_microbatches)],
            ))
    raise NotImplementedError('The default split_fn is unable to split the output of this'
                              'dataloader. Please define a split_fn in your dataloader spec.')


class DataloaderSpec(NamedTuple):
    dataset: Dataset
    drop_last: bool
    shuffle: bool
    collate_fn: Optional[Callable[[List[Any]], Batch]] = None
    worker_init_fn: Optional[Callable[[int], None]] = None
    multiprocessing_context: Any = None
    generator: Optional[torch.Generator] = None
    prefetch_fn: Optional[TPrefetchFn] = None
    split_fn: Callable[[Batch, int], List[Batch]] = _split_fn


@dataclass
class DatasetHparams(hp.Hparams, ABC):
    pass

    @abstractmethod
    def initialize_object(self) -> DataloaderSpec:
        pass
