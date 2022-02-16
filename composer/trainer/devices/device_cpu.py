# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Union
from packaging import version

import torch
from composer.core.types import Precision, StateDict, Tensor
from composer.trainer.devices.device import Device, T_nnModule

logger = logging.getLogger(__name__)


class DeviceCPU(Device):
    """An extension of :class:`~composer.trainer.devices.device.Device` for CPUs."""

    dist_backend = "gloo"

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module

    def tensor_to_device(self, tensor: Tensor) -> Tensor:
        return tensor

    @contextmanager
    def precision_context(self, precision: Union[str, Precision]) -> Generator[None, None, None]:
        precision = Precision(precision)
        if precision == Precision.FP32:
            yield
        elif precision == Precision.BF16:
            if version.parse(torch.__version__) < version.parse("1.10"):
                raise ValueError("Bfloat16 is only available for PyTorch versions >= 1.10")
            with torch.autocast(device_type="cpu", enabled=True, dtype=torch.bfloat16):
                yield
        else:
            raise ValueError(f"Precision {precision} not supported for a CPU")

    def state_dict(self) -> StateDict:
        # CPU device has no RNG state
        return {}

    def load_state_dict(self, state: StateDict) -> None:
        if len(state) != 0:
            raise ValueError("CPU device has no state.")
