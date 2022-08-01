# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Dataloader and Dataset Hyperparameter classes."""

from __future__ import annotations

import abc
import logging
import textwrap
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

import torch
import torch.distributed
import torch.utils.data
import yahp as hp

from composer.core.data_spec import DataSpec
from composer.core.types import Dataset

log = logging.getLogger(__name__)

__all__ = ['DataLoaderHparams']


@dataclass
class DataLoaderHparams(hp.Hparams):
    """Hyperparameters to initialize a :class:`torch.utils.data.DataLoader`.

    Args:
        num_workers (int, optional): Number of CPU workers to use per device to fetch data.
            Set to ``0`` to use the main training thread for dataloading.
            While zero workers can be useful for debugging, it should not be used for performance reasons.
            Default: ``8``.
        prefetch_factor (int, optional): Number of samples loaded in advance by each worker.
            For example, 2 means there will be a total of 2 * num_workers samples prefetched across all workers.
            If ``num_workers = 0``, then the ``prefetch_factor`` must be left at the default value.
            Default: ``2``.
        persistent_workers (bool): Whether to reuse dataloader workers across epochs. If ``num_workers`` is 0,
            then this field must be ``False``. Default: ``True``.
        pin_memory (bool, optional): Whether or not to copy Tensors into CUDA pinned memory before returning them.
            If ``num_workers = 0``, then the ``pin_memory`` must be ``False``. Default: ``True``.
        timeout (float): Timeout, in seconds, for collecting a batch from workers. Set to ``0`` for no timeout.
            Default: ``0``.
    """

    num_workers: int = hp.optional(textwrap.dedent("""\
        Number of CPU workers to use per device to fetch data.
        Set to ``0`` to use the main training thread for dataloading.
        While zero workers can be useful for debugging, it should not be used for performance reasons."""),
                                   default=8)
    prefetch_factor: int = hp.optional(textwrap.dedent("""\
        Number of samples loaded in advance by each worker.
        For example, 2 means there will be a total of 2 * num_workers samples prefetched across all workers.
        If ``num_workers = 0``, then the ``prefetch_factor`` must be left at the default value."""),
                                       default=2)
    persistent_workers: bool = hp.optional(textwrap.dedent("""\
         Whether to reuse dataloader workers across epochs. If ``num_workers`` is 0,
            then this field must be ``False``"""),
                                           default=True)
    pin_memory: bool = hp.optional(textwrap.dedent("""\
            Whether or not to copy Tensors into CUDA pinned memory before returning them.
            If ``num_workers = 0``, then the ``pin_memory`` must be ``False``."""),
                                   default=True)
    timeout: float = hp.optional(
        'Timeout, in seconds, for collecting a batch from workers. Set to ``0`` for no timeout.', default=0.0)

    def initialize_object(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        sampler: Optional[torch.utils.data.Sampler[int]],
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
    ):
        """Create a dataloader.

        Args:
            dataset (Dataset): The dataset.
            batch_size (int): The per-device batch size.
            sampler (torch.utils.data.Sampler[int] | None): The sampler to use for the dataloader.
            drop_last (bool): Whether to drop the last batch if the number of
                samples is not evenly divisible by the batch size.
            collate_fn (callable, optional): Custom collate function. Default: ``None``.
            worker_init_fn (callable, optional): Custom worker init function. Default: ``None``.

        Returns:
            DataLoader: The dataloader.
        """

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            sampler=sampler,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            timeout=self.timeout,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )


@dataclass
class DatasetHparams(hp.Hparams, abc.ABC):
    """Abstract base class for hyperparameters to initialize a dataset.

    Args:
        drop_last (bool): If the number of samples is not divisible by the batch size,
            whether to drop the last batch or pad the last batch with zeros. Default:
            ``True``.
        shuffle (bool): Whether to shuffle the dataset. Default: ``True``.
    """

    drop_last: bool = hp.optional(textwrap.dedent("""\
        If the number of samples is not divisible by the batch size,
        whether to drop the last batch (the default) or pad the last batch with zeros."""),
                                  default=True)
    shuffle: bool = hp.optional('Whether to shuffle the dataset for each epoch. Defaults to True.', default=True)

    @abc.abstractmethod
    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> Union[Iterable, DataSpec]:
        """Creates a :class:`~.core.types.DataLoader` or
        :class:`~.core.data_spec.DataSpec` for this dataset.

        Args:
            batch_size (int): The size of the batch the dataloader should yield. This
                batch size is device-specific and already incorporates the world size.
            dataloader_hparams (DataLoaderHparams): The dataset-independent hparams for
                the dataloader.

        Returns:
            Iterable | DataSpec: An iterable that yields batches, or if the dataset yields batches that need custom
            processing, a :class:`~core.data_spec.DataSpec`.
        """
        pass
