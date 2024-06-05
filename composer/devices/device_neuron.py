# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The Neuron device used for training."""

from __future__ import annotations

import logging
import os
from typing import TypeVar

import torch

from composer.devices.device import Device

logger = logging.getLogger(__name__)

__all__ = ['DeviceNeuron']

T_nnModule = TypeVar('T_nnModule', bound=torch.nn.Module)


class DeviceNeuron(Device):
    """An extension of :class:`~composer.devices.device.Device` for Neuron devices (Trn, Inf).

    When running on Trn, we automatically set `export PJRT_DEVICE=NEURON`.
    """

    name = 'neuron'
    dist_backend = 'xla'

    def __init__(self) -> None:
        import torch_xla.core.xla_model as xm

        # Turn off compiler based mixed precision (we use torch's amp support)
        # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/appnotes/neuronx-cc/neuronx-cc-training-mixed-precision.html
        os.environ['NEURON_CC_FLAGS'] = '--auto-cast=none'
        os.environ['PJRT_DEVICE'] = 'NEURON'
        self._device = xm.xla_device()

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)
