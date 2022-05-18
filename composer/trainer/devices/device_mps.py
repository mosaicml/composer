# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The GPU device used for training."""

from __future__ import annotations

from typing import Any, Dict, TypeVar

import torch
import torch.cuda.amp
import torch.utils.data

from composer.trainer.devices.device import Device

__all__ = ["DeviceMPS"]

T_nnModule = TypeVar("T_nnModule", bound=torch.nn.Module)


class DeviceMPS(Device):
    """An extension of :class:`~composer.trainer.devices.device.Device` for MPS,
    which is Apple's backend for training models on M1 chips.

    This class takes no arguments.
    """
    dist_backend = ""

    def __init__(self):
        self._device = torch.device("mps")

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device, non_blocking=True)

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if len(state) != 0:
            raise ValueError("MPS device has no state.")