# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Module for devices on which models run."""

from composer.devices.device import Device as Device
from composer.devices.device_cpu import DeviceCPU as DeviceCPU
from composer.devices.device_gpu import DeviceGPU as DeviceGPU
from composer.devices.device_mps import DeviceMPS as DeviceMPS
from composer.devices.device_tpu import DeviceTPU as DeviceTPU

__all__ = ['Device', 'DeviceCPU', 'DeviceGPU', 'DeviceMPS', 'DeviceTPU']
