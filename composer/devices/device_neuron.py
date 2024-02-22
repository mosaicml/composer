# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The Neuron device used for training."""

from __future__ import annotations

import logging
from typing import Any, Dict, TypeVar
import os
import torch

from composer.devices.device import Device

logger = logging.getLogger(__name__)

__all__ = ['DeviceNeuron']

T_nnModule = TypeVar('T_nnModule', bound=torch.nn.Module)


class DeviceNeuron(Device):
    """An extension of :class:`~composer.devices.device.Device` for Neuron devices (Trn, Inf).
    When running on Trn, we automatically set `export PJRT_DEVICE=NEURON`.
    More details.
    """

    name = 'neuron'
    dist_backend = 'xla'

    def __init__(self):
        import torch_xla.core.xla_model as xm
        os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"
        os.environ['PJRT_DEVICE']='NEURON'
        self._device = xm.xla_device()

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if len(state) != 0:
            raise ValueError('Neuron device has no state.')