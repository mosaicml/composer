# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The CPU device used for training."""

from __future__ import annotations

import logging
from typing import TypeVar

import torch

from composer.devices.device import Device

logger = logging.getLogger(__name__)

__all__ = ['DeviceCPU']

T_nnModule = TypeVar('T_nnModule', bound=torch.nn.Module)


class DeviceCPU(Device):
    """An extension of :class:`~composer.devices.device.Device` for CPUs.

    This class takes no arguments.
    """

    dist_backend = 'gloo'
    name = 'cpu'
    _device = torch.device('cpu')

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)
