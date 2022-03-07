# Copyright 2021 MosaicML. All Rights Reserved.

"""Specifications for operating and training on data."""
from __future__ import annotations

import collections.abc
import textwrap
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

import torch

from composer.utils.iter_helpers import ensure_tuple

if TYPE_CHECKING:
    from composer.core.types import Batch, DataLoader

__all__ = ["DataSpec"]


class DataSpec:
    """Specifications for operating and training on data.

    An example of constructing a :class:`DataSpec` object with a ``device_transforms`` callable
    (:class:`~composer.datasets.utils.NormalizationFn`) and then using it with :class:`~.Trainer`:

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
        dataloader (DataLoader): The dataloader.

        num_samples (int, optional): The total number of samples in an epoch, across all ranks. This field is used by
            the :class:`~.time.Timer` (training progress tracker). If not specified, then ``len(dataloader.dataset)`` is
            used (if this property is available). Otherwise, the dataset is assumed to be unsized.

        num_tokens (int, optional): The total number of tokens in an epoch. This field is used by the
            :class:`~.time.Timer` (training progress tracker).

        device_transforms ((Batch) -> Batch, optional): Function called by the :class:`~.trainer.Trainer` to modify the
            batch once it has been moved onto the device. For example, this function can be used for GPU-based
            normalization.  It can modify the batch in-place, and it should return the modified batch. If not specified, the
            batch is not modified.

        split_batch ((Batch, int) -> Sequence[Batch], optional): Function called by the :class:`~.trainer.Trainer` to
            split a batch (the first parameter) into the number of microbatches specified (the second parameter).  By
            default, batches of type :attr:`~.types.BatchPair` can be split automatically. If the ``dataloader`` yields
            batches of a different type, then this function must be specified.

        get_num_samples_in_batch ((Batch) -> int, optional): Function that is called by the :class:`~.trainer.Trainer`
            to get the number of samples in the provided batch.

            By default, if the batch contains tensors that all have the same 0th dim, then the value of the 0th dim will
            be returned. If the batch contains tensors where the 0th dim differ, then this function must be specified.

        get_num_tokens_in_batch ((Batch) -> int, optional): Function that is called by the :class:`~.trainer.Trainer` to
            get the number of tokens in the provided batch.

            By default, it returns 0, meaning that number of tokens processed will not be tracked as a part of the
            training progress tracking. This function must be specified to track the number of tokens processed during
            training.
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
        if num_microbatches < 1:
            raise ValueError("num_microbatches must be at least 1")
        if num_microbatches == 1:
            return [batch]
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
                textwrap.dedent(f"""\
                    Cannot determine the batch size, as multiple Tensors of
                    different lengths were found in the batch: sizes in batch: {dim0_sizes}.
                    Please use a DataSpec and specify `get_num_samples_in_batch`."""))

    def _default_get_num_tokens_in_batch(self, batch: Batch) -> int:
        del batch  # unused
        return 0
