# Copyright 2021 MosaicML. All Rights Reserved.

"""The TPU device used for training."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, TypeVar, Union

import torch
import torch_xla

from composer.core.precision import Precision
from composer.trainer.devices.device import Device, T_nnModule

logger = logging.getLogger(__name__)

__all__ = ["DeviceTPU"]

T_nnModule = TypeVar("T_nnModule", bound=torch.nn.Module)


class DeviceTPU(Device):
    """An extension of :class:`~composer.trainer.devices.device.Device` for TPUs

    This class takes no arguments.
    """
    dist_backend = "xla-tpu"
    def __init__(self):
        import torch_xla.core.xla_model as xm

        self._device = xm.xla_device()

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        import torch_xla.distributed.xla_multiprocessing as xmp

        return module.to(self._device)
        #wrapped_model = xmp.MpModelWrapper(module)
        return wrapped_model.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)

    def state_dict(self) -> StateDict:
        return {}

    def load_state_dict(self, state: StateDict) -> None:
        if len(state) != 0:
            raise ValueError("TPU device has no state.")
