# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`~yahp.hparams.Hparams` used to construct devices."""

from __future__ import annotations

from typing import Dict, Type, Union

import yahp as hp

from composer.trainer.devices.device import Device
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU

__all__ = ['device_registry']

device_registry: Dict[str, Union[Type[Device], Type[hp.Hparams]]] = {
    'gpu': DeviceGPU,
    'cpu': DeviceCPU,
}
