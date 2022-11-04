# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The TPU device used for training."""

from __future__ import annotations

import logging
from typing import Any, Dict, TypeVar

import torch

from composer.devices.device import Device

logger = logging.getLogger(__name__)

__all__ = ['DeviceTPU']

T_nnModule = TypeVar('T_nnModule', bound=torch.nn.Module)


class DeviceTPU(Device):
    """An extension of :class:`~composer.devices.device.Device` for TPUs.

    When running on TPUVMs, you need to `export PJRT_DEVICE=TPU`.
    More details.
    """

    def __init__(self):
        import torch_xla.core.xla_model as xm

        self._device = xm.xla_device()

    def module_to_device(self, module: T_nnModule) -> T_nnModule:

        return module.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if len(state) != 0:
            raise ValueError('TPU device has no state.')
