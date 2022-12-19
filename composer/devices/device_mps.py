# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The Apple M-series device used for training."""

from __future__ import annotations

from typing import Any, Dict, TypeVar

import torch
import torch.cuda.amp
import torch.utils.data
from packaging import version

from composer.devices.device import Device

__all__ = ['DeviceMPS']

T_nnModule = TypeVar('T_nnModule', bound=torch.nn.Module)


class DeviceMPS(Device):
    """Device to support MPS, for training on Apple's M-series chips.

    This class takes no arguments.
    """
    dist_backend = ''

    def __init__(self):
        if version.parse(torch.__version__) < version.parse('1.12.0'):
            raise RuntimeError('Support for MPS device requires torch >= 1.12.')
        if not torch.backends.mps.is_available():  # type: ignore (version guarded)
            raise RuntimeError('MPS requires MAC OSX >= 12.3')
        if not torch.backends.mps.is_built():  # type: ignore (version guarded)
            raise RuntimeError('torch was not build with MPS support.')

        self._device = torch.device('mps')

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if len(state) != 0:
            raise ValueError('MPS device has no state.')
