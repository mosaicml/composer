# Copyright 2021 MosaicML. All Rights Reserved.

"""Module for devices on which models run.

Used by :class:`~composer.trainer.trainer.Trainer` in order to train on different devices such as GPU and CPU.
"""

from composer.trainer.devices.device import Device as Device
from composer.trainer.devices.device_cpu import DeviceCPU as DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU as DeviceGPU
from composer.trainer.devices.device_hparams import CPUDeviceHparams as CPUDeviceHparams
from composer.trainer.devices.device_hparams import DeviceHparams as DeviceHparams
from composer.trainer.devices.device_hparams import GPUDeviceHparams as GPUDeviceHparams

__all__ = ["Device", "DeviceCPU", "DeviceGPU"]
