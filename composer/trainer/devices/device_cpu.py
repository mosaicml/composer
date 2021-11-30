# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional, Union

import torch

from composer.core.types import Batch, DataLoader, Precision, StateDict, Tensor, TPrefetchFn
from composer.datasets.dataloader import WrappedDataLoader
from composer.trainer.devices.device import Device, T_nnModule

logger = logging.getLogger(__name__)


class PrefetchedDataLoader(WrappedDataLoader):

    def __init__(self, dataloader: DataLoader, batch_preprocessing_fn: Optional[TPrefetchFn]) -> None:
        super().__init__(dataloader)
        self.device = torch.device("cpu")
        self.batch_preprocessing_fn = batch_preprocessing_fn

    def __iter__(self) -> Generator[Batch, None, None]:
        for batch in self.dataloader:
            if self.batch_preprocessing_fn is None:
                yield batch
            else:
                yield self.batch_preprocessing_fn(batch)


class DeviceCPU(Device):
    """An extension of :class:`~composer.trainer.devices.device.Device` for CPUs."""

    def prepare(self) -> None:
        logger.info("Preparing CPU worker")
        # No preperation required
        return

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module

    def dataloader_to_device(self, dataloader: DataLoader, prefetch_fn: Optional[TPrefetchFn]) -> DataLoader:
        return PrefetchedDataLoader(dataloader, prefetch_fn)

    def tensor_to_device(self, tensor: Tensor) -> Tensor:
        return tensor

    @contextmanager
    def precision_context(self, precision: Union[str, Precision]) -> Generator[None, None, None]:
        precision = Precision(precision)
        if precision == Precision.FP32:
            yield
        else:
            raise ValueError(f"Precision {precision} not supported for a CPU")

    @property
    def ddp_backend(self) -> str:
        return "gloo"

    def state_dict(self) -> StateDict:
        # CPU device has no RNG state
        return {}

    def load_state_dict(self, state: StateDict) -> None:
        if len(state) != 0:
            raise ValueError("CPU device has no state.")
