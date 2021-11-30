# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Union

from composer.core.types import Precision, StateDict, Tensor
from composer.trainer.devices.device import Device, T_nnModule

logger = logging.getLogger(__name__)


class DeviceCPU(Device):
    """An extension of :class:`~composer.trainer.devices.device.Device` for CPUs."""

    def prepare(self) -> None:
        logger.info("Preparing CPU worker")
        # No preperation required
        return

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module

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
