# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Module for devices on which models run.

Used by :class:`~composer.trainer.trainer.Trainer` in order to train on different devices such as GPU and CPU.
"""

from composer.trainer.devices.device import Device as Device
from composer.trainer.devices.device_cpu import DeviceCPU as DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU as DeviceGPU

__all__ = ['Device', 'DeviceCPU', 'DeviceGPU']
