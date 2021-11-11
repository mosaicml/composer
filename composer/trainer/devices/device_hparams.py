# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import yahp as hp

from composer.trainer.devices.device import Device
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU


@dataclass
class DeviceHparams(hp.Hparams):

    @abstractmethod
    def initialize_object(self) -> Device:
        pass


@dataclass
class GPUDeviceHparams(DeviceHparams):
    prefetch_in_cuda_stream: bool = hp.optional(doc="Whether to use a separate cuda stream for prefetching",
                                                default=True)

    def initialize_object(self) -> DeviceGPU:
        return DeviceGPU(prefetch_in_cuda_stream=self.prefetch_in_cuda_stream)


@dataclass
class CPUDeviceHparams(DeviceHparams):

    def initialize_object(self) -> DeviceCPU:
        return DeviceCPU()
