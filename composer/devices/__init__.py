# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Module for devices on which models run."""

from composer.devices.device import Device
from composer.devices.device_cpu import DeviceCPU
from composer.devices.device_gpu import DeviceGPU
from composer.devices.device_hpu import DeviceHPU
from composer.devices.device_mps import DeviceMPS
from composer.devices.device_neuron import DeviceNeuron
from composer.devices.device_tpu import DeviceTPU

__all__ = ['Device', 'DeviceCPU', 'DeviceGPU', 'DeviceMPS', 'DeviceNeuron', 'DeviceTPU', 'DeviceHPU']
