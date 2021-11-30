# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional, Union

import torch
import torch.cuda.amp
import torch.utils.data

from composer.core.types import Precision, StateDict, Tensor
from composer.trainer.devices.device import Device, T_nnModule
from composer.utils import ddp


class DeviceGPU(Device):
    """An extension of :class:`~composer.trainer.devices.device.Device` for GPUs.

    Args:
        prefetch_in_cuda_stream (bool): ``True`` to asyncrhonously prefetch
            samples with a CUDA stream during dataloading and ``False``
            otherwise.
    """

    def __init__(
        self,
        prefetch_in_cuda_stream: bool,
    ):
        self.prefetch_in_cuda_stream = prefetch_in_cuda_stream
        self._device: Optional[torch.device] = None

    def prepare(self) -> None:
        if self._device is not None:
            raise ValueError("device is already set")
        gpu = ddp.get_local_rank()
        self._device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(self._device)
        assert torch.cuda.current_device() == gpu

    def _require_device(self) -> None:
        if self._device is None:
            raise RuntimeError("device.module_to_device() was called before device.prepare(). "
                               "Please call device.prepare() first.")

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        self._require_device()
        assert self._device is not None
        return module.to(self._device)

    def tensor_to_device(self, tensor: Tensor) -> Tensor:
        self._require_device()
        assert self._device is not None
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

    @property
    def ddp_backend(self) -> str:
        return "nccl"

    def state_dict(self) -> StateDict:
        return {
            "rng": torch.cuda.get_rng_state(),
        }

    def load_state_dict(self, state: StateDict) -> None:
        torch.cuda.set_rng_state(state["rng"])
