# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Device-related helper methods and utilities."""

from typing import TYPE_CHECKING, Optional, Union

import torch.cuda

if TYPE_CHECKING:
    from composer.devices import Device

__all__ = ['get_device', 'is_tpu_installed']


def get_device(device: Optional[Union[str, 'Device']]) -> 'Device':
    """Takes string or Device and returns the corresponding :class:`~composer.devices.Device`.

    Args:
        device (str | Device, optional): A string corresponding to a device (one of
            ``'cpu'``, ``'gpu'``, ``'mps'``, or ``'tpu'``) or a :class:`.Device`.

    Returns:
        Device: Device corresponding to the passed string or
            Device. If no argument is passed, returns :class:`.DeviceGPU` if available,
            or :class:`.DeviceCPU` if no GPU is available.
    """
    from composer.devices import DeviceCPU, DeviceGPU, DeviceMPS, DeviceTPU

    if not device:
        device = DeviceGPU() if torch.cuda.is_available() else DeviceCPU()
    elif isinstance(device, str):
        if device.lower() == 'cpu':
            device = DeviceCPU()
        elif device.lower() == 'gpu':
            device = DeviceGPU()
        elif device.lower() == 'mps':
            device = DeviceMPS()
        elif device.lower() == 'tpu':
            if not is_tpu_installed():
                raise ImportError(
                    'Unable to import torch_xla. Please follow installation instructions at https://github.com/pytorch/xla'
                )
            device = DeviceTPU()
        else:
            raise ValueError(f'device ({device}) must be one of (cpu, gpu, mps, tpu).')
    return device


def is_tpu_installed() -> bool:
    """Determines whether the module needed for training on TPUs—torch_xla—is installed.

    Returns:
        bool: Whether torch_xla is installed.
    """
    try:
        import torch_xla
        del torch_xla
        return True
    except ModuleNotFoundError:
        return False
