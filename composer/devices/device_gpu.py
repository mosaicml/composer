# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The GPU device used for training."""

from __future__ import annotations

from typing import Any, Dict, Optional, TypeVar

import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.cuda
import torch.cuda.amp
import torch.utils.data

from composer.devices.device import Device
from composer.utils import dist

__all__ = ['DeviceGPU']

T_nnModule = TypeVar('T_nnModule', bound=torch.nn.Module)


class DeviceGPU(Device):
    """An extension of :class:`~composer.devices.device.Device` for GPUs.

    Args:
        device_id (int, optional): Integer ID of a GPU device to train with. If not specified, the local rank
            of the current process is used. Default: None.
        allow_tf32 (bool, optional): Whether to allow TF32 matrix multiplications. Defaults to True.
            For more information, see :ref:`torch:tf32_on_ampere`.
    """
    dist_backend = 'nccl'

    def __init__(
        self,
        device_id: Optional[int] = None,
        *,
        allow_tf32: bool = True,
    ):
        if not torch.cuda.is_available():
            raise ValueError('DeviceGPU cannot be created as torch.cuda is not available.')
        if device_id is None:
            device_id = dist.get_local_rank()
        self._device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(self._device)
        assert torch.cuda.current_device() == device_id
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        # pyright error: "allow_tf32" is not a known member of module
        # however, this flag exists on pytorch 1.9+: https://pytorch.org/docs/1.9.0/backends.html
        torch.backends.cudnn.allow_tf32 = allow_tf32  # type: ignore

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
