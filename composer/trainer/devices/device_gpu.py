# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The GPU device used for training."""

from __future__ import annotations

from typing import Any, Dict, Optional, TypeVar

import torch
import torch.cuda.amp
import torch.utils.data

from composer.trainer.devices.device import Device
from composer.utils import dist

__all__ = ['DeviceGPU']

T_nnModule = TypeVar('T_nnModule', bound=torch.nn.Module)


class DeviceGPU(Device):
    """An extension of :class:`~composer.trainer.devices.device.Device` for GPUs.

    Args:
        device_id (int, optional): Integer ID of a GPU device to train with. If not specified, the local rank
        of the current process is used. Default: None.
    """
    dist_backend = 'nccl'

    def __init__(self, device_id: Optional[int] = None):
        if not device_id:
            device_id = dist.get_local_rank()
        self._device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(self._device)
        assert torch.cuda.current_device() == device_id

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device, non_blocking=True)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'rng': torch.cuda.get_rng_state(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        torch.cuda.set_rng_state(state['rng'])
