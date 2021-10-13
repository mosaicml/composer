# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import multiprocessing
from abc import abstractmethod
from dataclasses import dataclass

import torch
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
    n_gpus: int = hp.required(doc="Number of GPUs to use per node. Set to -1 to use all.", template_default=-1)
    prefetch_in_cuda_stream: bool = hp.optional(doc="Whether to use a separate cuda stream for prefetching",
                                                default=True)

    def initialize_object(self) -> DeviceGPU:
        if self.n_gpus == -1:
            self.n_gpus = torch.cuda.device_count()
        if self.n_gpus < 1:
            raise ValueError("When using a GPU device, n_gpus must be at least 1.")
        return DeviceGPU(n_gpus=self.n_gpus, prefetch_in_cuda_stream=self.prefetch_in_cuda_stream)


@dataclass
class CPUDeviceHparams(DeviceHparams):
    n_cpus: int = hp.required(doc="Number of CPUs to use per node. Set to -1 to use multiprocessing.cpu_count()",
                              template_default=-1)

    def initialize_object(self) -> DeviceCPU:
        if self.n_cpus == -1:
            self.n_cpus = multiprocessing.cpu_count()
        if self.n_cpus < 1:
            raise ValueError("When using a CPU device, n_cpus must be at least 1.")
        return DeviceCPU(num_cpus=self.n_cpus)
