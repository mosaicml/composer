# Copyright 2021 MosaicML. All Rights Reserved.

import collections.abc
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from unittest import mock

import pytest
import torch
import torch.distributed
import yahp as hp
from _pytest.monkeypatch import MonkeyPatch

import composer.core.types as types
from composer import Callback
from composer.callbacks import CallbackHparams
from composer.core.logging import Logger
from composer.core.state import State
from composer.datasets import DataloaderHparams, DataloaderSpec, MemoryFormat, SyntheticDataset, SyntheticDatasetHparams
from composer.models.classify_mnist import MnistClassifierHparams
from composer.trainer.ddp import DDPHparams, FileStoreHparams
from composer.trainer.devices import CPUDeviceHparams, GPUDeviceHparams
from composer.trainer.trainer_hparams import TrainerHparams, callback_registry, dataset_registry


def get_file_path(tmpdir: str, *, idx: int, epoch: int, is_train: bool) -> str:
    train_str = "train" if is_train else "val"
    return os.path.join(tmpdir, f"{train_str}-epoch-{epoch}-sample-{idx}")


def get_batch_file_path(tmpdir: str, *, rank: int, epoch: int, is_train: bool) -> str:
    train_str = "train" if is_train else "val"
    return os.path.join(tmpdir, f"{train_str}-rank-{rank}-epoch-{epoch}-batch0.pt")


class TrackedDataset(SyntheticDataset):
    """
    TrackedDataset atomically writes a file every time a record is accessed.
    It is thread-safe and subprocess-safe, and is useful to measure how many times a sample is accessed.
    Because of atomic file writes, it is slow and should not be used in any performance measurements.
    """

    def __init__(self, *, sample_pool_size: int, shape: Sequence[int], memory_format: MemoryFormat, device: str,
                 one_hot: bool, num_classes: int, is_train: bool, tmpdir: str):
        super().__init__(sample_pool_size=sample_pool_size,
                         shape=shape,
                         memory_format=memory_format,
                         device=device,
                         one_hot=one_hot,
                         num_classes=num_classes)
        self.is_train = is_train
        self.tmpdir = tmpdir

    def __getitem__(self, idx: int):
        access = 0
        while True:
            try:
                with open(get_file_path(self.tmpdir, idx=idx, epoch=access, is_train=self.is_train), "x") as f:
                    f.write(str(idx))
                return super().__getitem__(idx)
            except FileExistsError:
                access += 1


@dataclass
class TrackedDatasetHparams(SyntheticDatasetHparams):
    is_train: Optional[bool] = hp.optional("is_train", default=None)
    tmpdir: Optional[str] = hp.optional("tmpdir", default=None)

    def initialize_object(self) -> DataloaderSpec:
        assert self.is_train is not None
        assert self.tmpdir is not None
        return DataloaderSpec(
            TrackedDataset(
                num_classes=self.num_classes,
                shape=self.shape,
                one_hot=self.one_hot,
                device=self.device,
                memory_format=self.memory_format,
                sample_pool_size=self.sample_pool_size,
                is_train=self.is_train,
                tmpdir=self.tmpdir,
            ),
            drop_last=self.drop_last,
            shuffle=self.shuffle,
        )


class CheckBatch0(Callback):

    def __init__(self, tmpdir: str):
        super().__init__()
        self.tmpdir = tmpdir

    def before_forward(self, state: State, logger: Logger):
        if state.batch_idx > 0:
            return
        rank: int = torch.distributed.get_rank()
        last_input, last_target = state.batch_pair
        torch.save(  # type: ignore
            {
                "last_input": last_input,
                "last_target": last_target,
            }, get_batch_file_path(self.tmpdir, rank=rank, epoch=state.epoch, is_train=True))

    def eval_before_forward(self, state: State, logger: Logger):
        rank: int = torch.distributed.get_rank()
        filepath = get_batch_file_path(self.tmpdir, rank=rank, epoch=state.epoch, is_train=False)
        if os.path.exists(filepath):
            return
        assert not state.model.training
        last_input, last_target = state.batch_pair
        torch.save(  # type: ignore
            {
                "last_input": last_input,
                "last_target": last_target,
            }, get_batch_file_path(self.tmpdir, rank=rank, epoch=state.epoch, is_train=False))


@dataclass
class CheckBatch0Hparams(CallbackHparams):
    tmpdir: str = hp.required("tmpdir")

    def initialize_object(self) -> Callback:
        return CheckBatch0(self.tmpdir)


@pytest.fixture(autouse=True)
def patch_registries(monkeypatch: MonkeyPatch):
    monkeypatch.setitem(callback_registry, "checkbatch0", CheckBatch0Hparams)
    monkeypatch.setitem(dataset_registry, "tracked", TrackedDatasetHparams)


@pytest.mark.run_long
@pytest.mark.timeout(90)
@pytest.mark.parametrize("fork_rank_0", [True, False], ids=["fork-rank-0", "no-fork-rank-0"])
@pytest.mark.parametrize("is_gpu,num_procs", [
    pytest.param(False, 1, id="1-cpu"),
    pytest.param(False, 2, id="2-cpu"),
    pytest.param(True, 1, marks=[pytest.mark.n_gpus(1)], id="1-gpu"),
    pytest.param(True, 2, marks=[pytest.mark.n_gpus(2)], id="2-gpu"),
])
def test_ddp(is_gpu: bool, num_procs: int, fork_rank_0: bool, *, ddp_tmpdir: str, is_main_pytest_process: bool,
             mosaic_trainer_hparams: TrainerHparams) -> None:
    """
    test strategy for ddp:
    1) Train a dummy model on two gps, for two epochs, using the tracked dataset.
    2) The tracked dataset should record two -- and only two -- accesses for each sample -- one for each epoch
       If each sample is accessed more than this number of times, then the distributed sampler isn't working properly
       If each sample is accessed less than this number of times, then either the sample pool size isn't a multiple of
       the batch size (and samples are getting dropped), or not all processes are working
    3) We use a callback to save the (x, y) for the first batch in each epoch on each process
        ({train, eval} * {epoch 1, epoch 2} * {ddp 1, ddp2})
       We assert that each of these tensors are different to ensure that 1) random seeding works properly,
       and 2) each ddp process is indeed getting different data.
    """
    hparams = mosaic_trainer_hparams
    assert isinstance(hparams, TrainerHparams)
    hparams.model = MnistClassifierHparams(num_classes=10)
    mosaic_trainer_hparams.train_dataset = TrackedDatasetHparams(
        num_classes=10,
        shape=[1, 28, 28],
        sample_pool_size=300,
        one_hot=False,
        device="cpu",
        is_train=True,
        memory_format=MemoryFormat.CONTIGUOUS_FORMAT,
        tmpdir=ddp_tmpdir,
    )
    hparams.val_dataset = TrackedDatasetHparams(
        num_classes=10,
        shape=[1, 28, 28],
        sample_pool_size=300,
        one_hot=False,
        device="cpu",
        is_train=False,
        memory_format=MemoryFormat.CONTIGUOUS_FORMAT,
        tmpdir=ddp_tmpdir,
    )
    if is_gpu:
        device = GPUDeviceHparams(n_gpus=num_procs)
    else:
        device = CPUDeviceHparams(n_cpus=num_procs)
    hparams.device = device
    hparams.ddp = DDPHparams(
        store=FileStoreHparams(os.path.join(ddp_tmpdir, "store")),
        node_rank=0,
        num_nodes=1,
        fork_rank_0=fork_rank_0,
    )
    hparams.dataloader = DataloaderHparams(
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        timeout=0,
    )
    hparams.total_batch_size = 50
    hparams.eval_batch_size = 50
    hparams.max_epochs = 2
    hparams.precision = types.Precision.FP32
    hparams.loggers = []
    hparams.validate_every_n_batches = 0
    hparams.validate_every_n_epochs = 1
    hparams.callbacks.append(CheckBatch0Hparams(tmpdir=ddp_tmpdir))
    trainer = hparams.initialize_object()
    assert trainer.state.world_size == num_procs
    assert trainer.state.nproc_per_node == num_procs
    assert isinstance(trainer.train_dl_spec.dataset, collections.abc.Sized)
    num_train_samples = len(trainer.train_dl_spec.dataset)
    assert isinstance(trainer.eval_dl_spec.dataset, collections.abc.Sized)
    num_eval_samples = len(trainer.eval_dl_spec.dataset)
    trainer.fit()

    # we want to validate on the spawning process only
    if is_main_pytest_process:
        # now validate that each sample were accessed exactly hparams.max_epochs * batch size times
        num_epochs = hparams.max_epochs

        for i in range(num_train_samples):
            for epoch in range(num_epochs):
                assert os.path.exists(
                    get_file_path(ddp_tmpdir, idx=i, epoch=epoch,
                                  is_train=True)), f"train sample {i} was not accessed during epoch {epoch}"
            assert not os.path.exists(get_file_path(ddp_tmpdir, idx=i, epoch=num_epochs,
                                                    is_train=True)), f"train sample {i} was accessed too many times"

        for i in range(num_eval_samples):
            for epoch in range(num_epochs):
                assert os.path.exists(
                    get_file_path(ddp_tmpdir, idx=i, epoch=epoch,
                                  is_train=False)), f"val sample {i} was not accessed during epoch {epoch}"
            # the eval dataloader is spun once more to initialize the rng, so expecting num_epochs + 1 to not exist
            assert not os.path.exists(get_file_path(ddp_tmpdir, idx=i, epoch=num_epochs + 1,
                                                    is_train=False)), f"val sample {i} was accessed too many times"

        is_train_to_pickles: Dict[bool, List[Dict[str, types.Tensor]]] = {True: [], False: []}

        for epoch in range(num_epochs):
            for local_rank in range(trainer.device.nproc_per_node):
                for is_train in (True, False):
                    data: Dict[str, types.Tensor] = torch.load(  # type: ignore
                        get_batch_file_path(ddp_tmpdir, rank=local_rank, epoch=epoch, is_train=is_train),
                        map_location='cpu',
                    )
                    for pickle in is_train_to_pickles[is_train]:
                        assert not torch.all(data['last_input'] == pickle['last_input'])
                        assert not torch.all(data['last_target'] == pickle['last_target'])
                    is_train_to_pickles[is_train].append(data)


def test_ddp_cuda_available_check(mosaic_trainer_hparams: TrainerHparams):
    with mock.patch.object(torch.cuda, 'device_count') as device_count, \
        mock.patch.object(torch.cuda, 'is_available') as is_cuda_available:
        is_cuda_available.return_value = False
        device_count = 1

        mosaic_trainer_hparams.device = GPUDeviceHparams(n_gpus=1)
        assert (not torch.cuda.is_available())

        with pytest.raises(ValueError, match="CUDA not available but gpu backend requested."):
            mosaic_trainer_hparams.initialize_object()


def test_ddp_cuda_ngpus_check(mosaic_trainer_hparams: TrainerHparams):
    with mock.patch.object(torch.cuda, 'device_count') as device_count, \
        mock.patch.object(torch.cuda, 'is_available') as is_cuda_available:
        is_cuda_available.return_value = True
        device_count.return_value = 2

        mosaic_trainer_hparams.device = GPUDeviceHparams(n_gpus=8)

        with pytest.raises(ValueError, match="Requested 8 GPUs, but only 2 available."):
            mosaic_trainer_hparams.initialize_object()


def test_ddp_nccl_check(mosaic_trainer_hparams: TrainerHparams):
    with mock.patch.object(torch.cuda, 'device_count') as device_count, \
        mock.patch.object(torch.distributed, 'is_nccl_available') as nccl_available, \
        mock.patch.object(torch.cuda, 'is_available') as is_cuda_available:

        device_count.return_value = 1
        is_cuda_available.return_value = True
        nccl_available.return_value = False

        mosaic_trainer_hparams.device = GPUDeviceHparams(n_gpus=1)

        with pytest.raises(ValueError, match="Requested NCCL backend not available in torch.distributed"):
            mosaic_trainer_hparams.initialize_object()
