# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import collections.abc
import logging
import os
import subprocess
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Thread
from typing import Callable, Iterator, List, Optional, Sequence, TypeVar, cast

import torch
import torch.distributed
import torch.utils.data
import yahp as hp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from composer.core.state import State
from composer.core.types import Batch, DataLoader, Model, Tensor
from composer.datasets import DataloaderHparams, DataloaderSpec, WrappedDataLoader

logger = logging.getLogger(__name__)

TObj = TypeVar("TObj")


class DataloaderMultipleIterationWarning(Warning):
    pass


class DDPDataLoader(WrappedDataLoader):
    """
    DDPDataLoader wraps a dataloader and a distributed sampler to ensure that
    sampler.set_epoch() is called after each iteration (epoch) through the dataset
    See https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
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


class DDP:

    def __init__(self,
                 *,
                 nproc_per_node: int,
                 store_hparams: StoreHparams,
                 node_rank: int,
                 num_nodes: int,
                 backend: str,
                 fork_rank_0: bool,
                 find_unused_parameters: bool = False):
        self.nproc_per_node = nproc_per_node
        self.world_size = num_nodes * nproc_per_node
        self.num_nodes = num_nodes
        self.node_rank = node_rank
        self.store_hparams = store_hparams
        self.last_return_code: Optional[int] = None
        self.backend = backend
        self.fork_rank_0 = fork_rank_0
        self.processes: List[subprocess.Popen[str]] = []
        self.find_unused_parameters = find_unused_parameters

        if backend == 'nccl':
            if not torch.cuda.is_available():
                raise ValueError('CUDA not available but gpu backend requested.')
            if torch.cuda.device_count() < nproc_per_node:
                raise ValueError(f'Requested {nproc_per_node} GPUs, but '\
                                 f'only {torch.cuda.device_count()} available.')
            if not torch.distributed.is_nccl_available():
                raise ValueError('Requested NCCL backend not available in torch.distributed')

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

    def launch(self, state: State, loop: Callable[[], None]):
        if os.environ.get("RANK") is None:
            os.environ["WORLD_SIZE"] = str(self.world_size)
            logger.info("Starting DDP on node_rank(%d) with world_size(%d)", self.node_rank, self.world_size)

            if torch.distributed.is_available():
                # Adapted from torch.distributed.launch

                # set PyTorch distributed related environmental variables

                current_env = os.environ.copy()
                # TODO omp num threads -- this parameter needs to be auto-tuned
                for local_rank in range(self.nproc_per_node):
                    # each process's rank
                    global_rank = self.nproc_per_node * self.node_rank + local_rank
                    current_env["RANK"] = str(global_rank)

                    if local_rank == 0 and not self.fork_rank_0:
                        os.environ["RANK"] = str(global_rank)
                    else:
                        logger.info("Launching process for global_rank(%d) on node_rank(%d)", global_rank,
                                    self.node_rank)
                        # spawn the processes
                        cmd = [
                            sys.executable,
                            "-u",
                            *sys.argv,
                        ]

                        if local_rank == 0:
                            # Attaching rank 0 to the main stdout/stderr so interactive
                            # terminal output will work without issue (e.g. tqdm)
                            process = subprocess.Popen(cmd, env=current_env, text=True)
                        else:
                            # Other processes, except in the case of an error, should not print anything
                            process = subprocess.Popen(
                                cmd,
                                env=current_env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                            )
                        self.processes.append(process)
                if self.fork_rank_0:
                    self.monitor()
                    return
                else:
                    Thread(target=self.monitor, daemon=True).start()
            else:
                if self.world_size != 1:
                    raise ValueError("Must have world size == 1 when torch.distributed is not available")
                if self.node_rank != 0:
                    raise ValueError("Must have a node_rank == 0 when torch.distributed is not available")
                os.environ["RANK"] = "0"
        # We are now on the correct process
        global_rank = int(os.environ["RANK"])
        assert global_rank // self.world_size == self.node_rank
        assert os.environ["WORLD_SIZE"] == str(
            self.world_size
        ), f"os.environ['WORLD_SIZE']({os.environ['WORLD_SIZE']}) != self.world_size({self.world_size})"
        is_main = global_rank == 0
        if torch.distributed.is_available():
            logger.info("Initializing ddp: GLOBAL_RANK: %s, WORLD_SIZE: %s", global_rank, self.world_size)
            store = self.store_hparams.initialize_object(is_main, state.world_size)
            torch.distributed.init_process_group(self.backend,
                                                 rank=global_rank,
                                                 world_size=self.world_size,
                                                 store=store)
            assert torch.distributed.is_initialized()
            assert state.is_rank_set, "state.is_rank_set should be set after torch.distributed is initialized"
            assert state.local_rank == global_rank % self.nproc_per_node, "state.local_rank is incorrect"
            assert state.nproc_per_node == self.nproc_per_node, "state.nproc_per_node is incorrect"
            assert state.global_rank == torch.distributed.get_rank(
            ), "state.global_rank != torch.distributed.get_rank()"
            logger.info("All DDP processes registered. world_size=%s.", self.world_size)
            logger.info("Starting process with global_rank=%s", global_rank)
        try:
            loop()
        finally:
            self.cleanup()

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

    def monitor(self) -> None:
        # Monitor checks whether any subprocesses have died unexpectedly
        alive_processes = set(self.processes)
        while len(alive_processes) > 0:
            finished_processes: List[subprocess.Popen[str]] = []
            for process in alive_processes:
                if process.poll() is None:
                    # the process is still running
                    continue
                else:
                    if process.returncode != 0:
                        if process.stdout is None:
                            output = ""
                        else:
                            output = process.stdout.read()

                        if process.stderr is None:
                            stderr = ""
                        else:
                            stderr = process.stderr.read()
                        exc = subprocess.CalledProcessError(
                            process.returncode,
                            cmd=process.args,
                            output=output,
                            stderr=stderr,
                        )
                        if self.fork_rank_0:
                            raise exc
                        else:
                            logger.exception("Error in subprocess", exc_info=exc)
                            sys.exit(1)
                    else:
                        # exited cleanly
                        finished_processes.append(process)
            alive_processes = set(alive_processes) - set(finished_processes)
            time.sleep(1)

    def cleanup(self) -> None:
        for process in self.processes:
            logger.info("Killing subprocess %s", process.pid)
            try:
                process.kill()
            except Exception:
                pass
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@dataclass
class StoreHparams(hp.Hparams, ABC):

    @abstractmethod
    def initialize_object(self, is_main: bool, world_size: int) -> torch.distributed.Store:
        pass


@dataclass
class TCPStoreHparams(StoreHparams):
    host_name: str = hp.optional(doc="Rank 0 address", default="127.0.0.1")
    port: int = hp.optional(doc="Rank 0 port", default=43297)

    def initialize_object(self, is_main: bool, world_size: int) -> torch.distributed.Store:
        return torch.distributed.TCPStore(self.host_name, self.port, world_size, is_main)


@dataclass
class FileStoreHparams(StoreHparams):
    file_name: str = hp.required(doc="Path to store file")

    def initialize_object(self, is_main: bool, world_size: int) -> torch.distributed.Store:
        return torch.distributed.FileStore(self.file_name, world_size)


@dataclass
class DDPHparams(hp.Hparams):
    hparams_registry = {
        "store": {
            "tcp": TCPStoreHparams,
            "file": FileStoreHparams,
        }
    }

    store: StoreHparams = hp.optional(doc="Store", default_factory=TCPStoreHparams)
    node_rank: int = hp.optional(doc="Node ID for multi-node training", default=0)
    num_nodes: int = hp.optional(doc="Number of nodes used for training", default=1)
    fork_rank_0: bool = hp.optional(
        doc="Whether to fork the local rank 0 process, or use the existing process for rank 0 training.",
        default=False,
    )

    def initialize_object(self, nproc_per_node: int, backend: str, find_unused_parameters: bool) -> DDP:
        return DDP(
            backend=backend,
            nproc_per_node=nproc_per_node,
            store_hparams=self.store,
            node_rank=self.node_rank,
            num_nodes=self.num_nodes,
            fork_rank_0=self.fork_rank_0,
            find_unused_parameters=find_unused_parameters,
        )
