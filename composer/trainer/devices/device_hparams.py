# Copyright 2021 MosaicML. All Rights Reserved.

"""The :class:`~yahp.hparams.Hparams` used to construct devices."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import yahp as hp

from composer.trainer.devices.device import Device
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU

__all__ = ["DeviceHparams", "CPUDeviceHparams", "GPUDeviceHparams"]


@dataclass
class DeviceHparams(hp.Hparams):

    @abstractmethod
    def initialize_object(self) -> Device:
        pass


@dataclass
class GPUDeviceHparams(DeviceHparams):

    def initialize_object(self) -> DeviceGPU:
        return DeviceGPU()


@dataclass
class CPUDeviceHparams(DeviceHparams):

    def initialize_object(self) -> DeviceCPU:
        return DeviceCPU()
