# Copyright 2021 MosaicML. All Rights Reserved.

"""The GPU device used for training."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, TypeVar, Union

import torch
import torch.cuda.amp
import torch.utils.data

from composer.core.types import Precision, StateDict, Tensor
from composer.trainer.devices.device import Device, T_nnModule
from composer.utils import dist

__all__ = ["DeviceGPU"]

T_nnModule = TypeVar("T_nnModule", bound=torch.nn.Module)


class DeviceGPU(Device):
    """An extension of :class:`~composer.trainer.devices.device.Device` for GPUs.

    This class takes no arguments.
    """
    dist_backend = "nccl"

    def __init__(self):
        gpu = dist.get_local_rank()
        self._device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(self._device)
        assert torch.cuda.current_device() == gpu

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module.to(self._device)

    def tensor_to_device(self, tensor: Tensor) -> Tensor:
        return tensor.to(self._device, non_blocking=True)

    @contextmanager
    def precision_context(self, precision: Union[str, Precision]) -> Generator[None, None, None]:
        precision = Precision(precision)
        if precision == Precision.FP32:
            enabled = False
        elif precision == Precision.AMP:
            enabled = True
        else:
            raise ValueError(f"Precision {precision} not supported for a GPU")
        with torch.cuda.amp.autocast(enabled):  #type: ignore
            yield

    def state_dict(self) -> StateDict:
        return {
            "rng": torch.cuda.get_rng_state(),
        }

    def load_state_dict(self, state: StateDict) -> None:
        torch.cuda.set_rng_state(state["rng"])
