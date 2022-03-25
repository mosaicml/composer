# Copyright 2021 MosaicML. All Rights Reserved.

"""The CPU device used for training."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, TypeVar, Union

import torch

from composer.core import Precision
from composer.trainer.devices.device import Device, T_nnModule

logger = logging.getLogger(__name__)

__all__ = ["DeviceCPU"]

T_nnModule = TypeVar("T_nnModule", bound=torch.nn.Module)


class DeviceCPU(Device):
    """An extension of :class:`~composer.trainer.devices.device.Device` for CPUs.

    This class takes no arguments.
    """

    dist_backend = "gloo"
    _device = torch.device('cpu')

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)

    @contextmanager
    def precision_context(self, precision: Union[str, Precision]) -> Generator[None, None, None]:
        precision = Precision(precision)
        if precision == Precision.FP32:
            yield
        else:
            raise ValueError(f"Precision {precision} not supported for a CPU")

    def state_dict(self) -> Dict[str, Any]:
        # CPU device has no RNG state
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if len(state) != 0:
            raise ValueError("CPU device has no state.")
