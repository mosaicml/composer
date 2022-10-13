# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Module for devices on which models run.

Used by :class:`.Trainer` in order to train on different devices such as GPU and CPU.
"""

from composer.trainer.devices.device import Device as Device
from composer.trainer.devices.device_cpu import DeviceCPU as DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU as DeviceGPU
from composer.trainer.devices.device_mps import DeviceMPS as DeviceMPS
from composer.trainer.devices.device_tpu import DeviceTPU as DeviceTPU
from composer.trainer.devices.utils import get_device as get_device
from composer.trainer.devices.utils import is_tpu_installed as is_tpu_installed

__all__ = ['Device', 'DeviceCPU', 'DeviceGPU', 'DeviceMPS', 'DeviceTPU', 'get_device', 'is_tpu_installed']
