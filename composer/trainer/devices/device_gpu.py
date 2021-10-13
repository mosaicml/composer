# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional, Union

import torch
import torch.cuda.amp
import torch.utils.data

from composer.core.state import State
from composer.core.types import Batch, DataLoader, Precision, StateDict, Tensor, TPrefetchFn
from composer.datasets.dataloader import WrappedDataLoader
from composer.trainer.devices.device import Device, T_nnModule
from composer.utils import map_collection


class CudaDataLoader(WrappedDataLoader):
    """Wraps :class:`~composer.core.types.DataLoader` and moves samples onto
    the specified device as they are used.

    Args:
        dataloader (Dataloader): The dataloader to wrap.
        prefetch_in_cuda_stream (bool): ``True`` to asyncrhonously prefetch
            samples with a CUDA stream during dataloading and ``False``
            otherwise.
        device (:class:`torch.device`): The device that samples should
            automatically be moved to upon iteration.
        prefetch_fn (TPrefetchFn, optional): A function to run on the data
            after fetching it. (default: ``None``)
    """

    def __init__(
        self,
        dataloader: DataLoader,
        prefetch_in_cuda_stream: bool,
        device: torch.device,
        prefetch_fn: Optional[TPrefetchFn] = None,
    ) -> None:
        super().__init__(dataloader)
        self.device = device
        self.prefetch_in_cuda_stream = prefetch_in_cuda_stream
        self.prefetch_fn = prefetch_fn

    def __iter__(self):
        if self.prefetch_in_cuda_stream:
            return self.prefetch_in_cuda_stream_iter()
        else:
            return self.normal_iter()

    def normal_iter(self):
        """Iterate over data without prefetching."""
        for batch in self.dataloader:
            batch = self.move_to_gpu(batch)
            if self.prefetch_fn is not None:
                batch = self.prefetch_fn(batch)
            yield batch

    def prefetch_in_cuda_stream_iter(self):
        """Iterate over data while prefetching data in a cuda stream."""
        stream = torch.cuda.Stream()
        batch: Optional[Batch] = None

        for next_batch in self.dataloader:
            with torch.cuda.stream(stream):
                next_batch = self.move_to_gpu(next_batch)
                if self.prefetch_fn is not None:
                    next_batch = self.prefetch_fn(next_batch)
            if batch is not None:
                yield batch

            torch.cuda.current_stream().wait_stream(stream)
            batch = next_batch
        if batch is not None:
            yield batch

    def _to_device(self, x: Tensor) -> Tensor:
        return x.to(self.device, non_blocking=True)

    def move_to_gpu(self, batch: Batch) -> Batch:
        """Move data to the GPU device.

        Args:
            batch (Batch): The data to move the gpu.
        """
        if isinstance(batch, Tensor):
            return self._to_device(batch)
        if isinstance(batch, (tuple, list)):
            return type(batch)(self._to_device(x) for x in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}

        return map_collection(batch, self._to_device)  # type: ignore


class DeviceGPU(Device):
    """An extension of :class:`~composer.trainer.devices.device.Device` for GPUs.

    Args:
        prefetch_in_cuda_stream (bool): ``True`` to asyncrhonously prefetch
            samples with a CUDA stream during dataloading and ``False``
            otherwise.
        num_gpus (int): The number of GPUs to use.
    """

    def __init__(
        self,
        prefetch_in_cuda_stream: bool,
        n_gpus: int,
    ):
        self.n_gpus = n_gpus
        self.prefetch_in_cuda_stream = prefetch_in_cuda_stream
        self._device: Optional[torch.device] = None

    @property
    def nproc_per_node(self) -> int:
        return self.n_gpus

    def prepare(self, state: State) -> None:
        if self._device is not None:
            raise ValueError("device is already set")
        gpu = state.local_rank
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

    def dataloader_to_device(self, dataloader: DataLoader, prefetch_fn: Optional[TPrefetchFn]) -> DataLoader:
        self._require_device()
        assert self._device is not None
        return CudaDataLoader(
            dataloader,
            device=self._device,
            prefetch_in_cuda_stream=self.prefetch_in_cuda_stream,
            prefetch_fn=prefetch_fn,
        )

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

    @property
    def ddp_backend(self) -> str:
        return "nccl"

    def state_dict(self) -> StateDict:
        return {
            "rng": torch.cuda.get_rng_state(),
        }

    def load_state_dict(self, state: StateDict) -> None:
        torch.cuda.set_rng_state(state["rng"])
