# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import collections.abc
import datetime
import logging
import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, ContextManager, Iterator, List, Optional, Sequence, TypeVar, cast

import torch
import torch.distributed
import torch.utils.data
import yahp as hp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from composer.core.state import State
from composer.core.types import Batch, DataLoader, Model, Tensor
from composer.datasets import DataloaderHparams, DataloaderSpec, WrappedDataLoader
from composer.utils.ddp import get_world_size
from composer.utils.iter_helpers import ensure_tuple
from composer.utils.string_enum import StringEnum

logger = logging.getLogger(__name__)

TObj = TypeVar("TObj")

CLEANUP_TIMEOUT = datetime.timedelta(seconds=5)


class DataloaderMultipleIterationWarning(Warning):
    pass


class DDPDataLoader(WrappedDataLoader):
    """Ensure sampler.set_epoch() is called after each iteration.

    DDPDataLoader wraps a dataloader and a distributed sampler and is
    called after each iteration (epoch) through the dataset.
    See: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    """

    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__(dataloader)
        if not isinstance(self.dataloader.sampler, DistributedSampler):
            raise ValueError("When using the DDP data loader, the sampler must be a DistributedSampler")
        self._iterator: Optional[Iterator[Batch]] = None

    def __iter__(self) -> DDPDataLoader:
        if self._iterator is not None:
            warnings.warn(
                "The dataloader detected the start of a new iteration before the previous iteration finished. "
                "The dataloader is skipping ahead to the start of the next epoch. "
                "Multiple simultaneous iterations through the DDP dataloader prohibited, since "
                "it automatically tracks the current epoch.",
                category=DataloaderMultipleIterationWarning)
            assert isinstance(self.sampler, DistributedSampler)
            self.sampler.set_epoch(epoch=self.sampler.epoch + 1)
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self) -> Batch:
        assert self._iterator is not None
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = None
            assert isinstance(self.sampler, DistributedSampler)
            self.sampler.set_epoch(epoch=self.sampler.epoch + 1)
            raise


class DDPSyncStrategy(StringEnum):
    """How and when DDP gradient synchronization should happen.

    Attributes:
        SINGLE_AUTO_SYNC: The default behavior for DDP. Gradients are synchronized as they
            computed, for only the final microbatch of a batch. This is the most efficient
            strategy, but can lead to errors when ``find_unused_parameters`` is set, since
            it is possible different microbatches may use different sets of parameters,
            leading to an incomplete sync.
        MULTI_AUTO_SYNC: The default behavior for DDP when ``find_unused_parameters`` is set.
            Gradients are synchronized as they are computed for all microbatches. This ensures
            complete synchronization, but is less efficient than :attr:`SINGLE_AUTO_SYNC`. This
            efficiency gap is usually small, as long as either DDP syncs are a small portion
            of the trainer's overall runtime, or the number of microbatches per batch is
            relatively small.
        FORCED_SYNC: Gradients are manually synchronized only after all gradients have been
            computed for the final microbatch of a batch. Like :attr:`MULTI_AUTO_SYNC`, this
            strategy ensures complete gradient synchronization, but this tends to be slower than
            :attr:`MULTI_AUTO_SYNC`. This is because ordinarily syncs can happen in parallel
            with the ``loss.backward()`` computation, meaning syncs can be mostly complete by
            the time that function finishes. However, in certain circumstances, syncs may take
            a very long time to complete - if there are also a lot of microbatches per batch,
            this strategy may be optimal.
    """
    SINGLE_AUTO_SYNC = "single_auto_sync"
    MULTI_AUTO_SYNC = "multi_auto_sync"
    FORCED_SYNC = "forced_sync"


class DDP:

    def __init__(self,
                 *,
                 backend: str,
                 timeout: float,
                 find_unused_parameters: bool = False,
                 sync_strategy: Optional[str] = None):
        self.backend = backend
        self.find_unused_parameters = find_unused_parameters
        if sync_strategy is None:
            self.sync_strategy = DDPSyncStrategy.SINGLE_AUTO_SYNC if not find_unused_parameters else DDPSyncStrategy.FORCED_SYNC
        else:
            self.sync_strategy = DDPSyncStrategy(sync_strategy)

        _timeout = datetime.timedelta(seconds=timeout)

        if torch.distributed.is_initialized():

            if not torch.distributed.get_backend() == self.backend.lower():
                raise RuntimeError(
                    f"The requested backend ({self.backend}) differs from the backend "
                    "of the current process group ({torch.distributed.get_backend()}). If you wish to change backends, "
                    "please restart the python process.")
            return

        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Assume we can initialize based off of env vars
            torch.distributed.init_process_group(self.backend, timeout=_timeout)
            return

        warnings.warn("NoDDPWarning: RANK and WORLD_SIZE env vars not set; assuming no parallelization. "
                      "If this is unexpected, make sure you are running your training script with the "
                      "composer executable.")
        store = torch.distributed.HashStore()

        torch.distributed.init_process_group(self.backend, timeout=_timeout, store=store, world_size=1, rank=0)

    @property
    def world_size(self) -> int:
        return get_world_size()

    def barrier(self) -> None:
        if torch.distributed.is_available():
            torch.distributed.barrier()
        # If not on DDP, then do nothing

    def all_reduce(
        self,
        tensor: torch.Tensor,
        reduce_operation: str = "SUM",
    ) -> None:
        if torch.distributed.is_available():
            reduce_op = getattr(torch.distributed.ReduceOp, reduce_operation.upper())
            torch.distributed.all_reduce(tensor, op=reduce_op)
        else:
            raise NotImplementedError("Non-DDP versions of reduce operations are not yet implemented")

    def all_gather(self, tensor: torch.Tensor) -> Sequence[Tensor]:
        """gather_to_rank_zero collects a tensor from each rank, and returns a sequence of tensors indexed by rank

        Args:
            tensor (torch.Tensor): tensor from each rank to be gathered

        Returns:
            Sequence[Tensor]: A sequence of tensors indexed by rank
        """
        if torch.distributed.is_available():
            obj_gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            torch.distributed.all_gather(obj_gather_list, tensor)
            return obj_gather_list
        else:
            return [tensor]

    def all_gather_object(self, obj: TObj) -> List[TObj]:
        """gather_object_to_rank_zero collects a pickleable object from each rank, and returns a list of
        these objects indexed by rank

        Args:
            obj (TObj): Object to be gathered

        Returns:
            List[TObj]: A list of objects indexed by rank
        """
        if torch.distributed.is_available():
            obj_gather_list = [None for _ in range(self.world_size)]
            torch.distributed.all_gather_object(obj_gather_list, obj)
            # torch.distributed will replace the None's in obj_gather_list with the gathered objects on rank 0
            # or will just be None on non-rank-0
            return cast(List[TObj], obj_gather_list)
        else:
            return [obj]

    def prepare_module(self, module: Model) -> Model:
        if torch.distributed.is_available():
            if any((p.requires_grad for p in module.parameters())):
                ddp_model = DistributedDataParallel(module, find_unused_parameters=self.find_unused_parameters)
                return cast(Model, ddp_model)
            return module
        else:
            return module

    def create_dataloader(self, batch_size: int, dataloader_hparams: DataloaderHparams,
                          dataloader_spec: DataloaderSpec) -> DataLoader:
        if torch.distributed.is_available():
            sampler = torch.utils.data.DistributedSampler[int](dataloader_spec.dataset,
                                                               drop_last=dataloader_spec.drop_last,
                                                               shuffle=dataloader_spec.shuffle)
        else:
            assert isinstance(dataloader_spec.dataset, collections.abc.Sized)
            sampler = torch.utils.data.RandomSampler(dataloader_spec.dataset, generator=dataloader_spec.generator)
        dataloader = dataloader_hparams.initialize_object(batch_size, sampler, dataloader_spec)
        if torch.distributed.is_available():
            dataloader = DDPDataLoader(dataloader)
        return dataloader

    @contextmanager
    def sync_context(self, state: State, is_final_microbatch: bool):
        assert isinstance(state.model, DistributedDataParallel), "state.model is not wrapped by DDP"
        assert state.optimizers is not None, "optimizers have not been initialized"

        no_sync_context = cast(Callable[[], ContextManager], state.model.no_sync)
        auto_sync_context = nullcontext

        if self.sync_strategy == DDPSyncStrategy.SINGLE_AUTO_SYNC:
            context = auto_sync_context if is_final_microbatch else no_sync_context
            with context():
                yield

        elif self.sync_strategy == DDPSyncStrategy.MULTI_AUTO_SYNC:
            with auto_sync_context():
                yield

        elif self.sync_strategy == DDPSyncStrategy.FORCED_SYNC:
            try:
                with no_sync_context():
                    yield
            finally:
                if is_final_microbatch:
                    for optimizer in ensure_tuple(state.optimizers):
                        for group in optimizer.param_groups:
                            for p in group["params"]:
                                if p.grad is not None:
                                    self.all_reduce(p.grad)
                                    p.grad = p.grad / state.world_size

        else:
            raise ValueError("Unknown sync strategy", self.sync_strategy)


@dataclass
class DDPHparams(hp.Hparams):
    sync_strategy: Optional[str] = hp.optional(
        doc="The strategy for synchronizing DDP. Default value ``None`` causes the "
        "trainer to auto-select a value depending on what algorithms are used.",
        default=None)
    timeout: float = hp.optional(doc="Timeout, in seconds, for initializing the DDP process group.", default=5.0)
