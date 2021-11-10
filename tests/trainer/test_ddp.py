# Copyright 2021 MosaicML. All Rights Reserved.

import collections.abc
import os
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

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
from composer.trainer.devices import CPUDeviceHparams, GPUDeviceHparams
from composer.trainer.trainer_hparams import TrainerHparams, callback_registry, dataset_registry
from tests.fixtures.ddp_fixtures import with_distributed
from tests.fixtures.models import SimpleBatchPairModelHparams


def get_file_path(tmpdir: Union[str, pathlib.Path], *, idx: int, epoch: int, is_train: bool) -> str:
    train_str = "train" if is_train else "val"
    return os.path.join(tmpdir, f"{train_str}-epoch-{epoch}-sample-{idx}")


def get_batch_file_path(tmpdir: Union[str, pathlib.Path], *, rank: int, epoch: int, is_train: bool) -> str:
    train_str = "train" if is_train else "val"
    return os.path.join(tmpdir, f"{train_str}-rank-{rank}-epoch-{epoch}-batch0.pt")


class TrackedDataset(SyntheticDataset):
    """
    TrackedDataset atomically writes a file every time a record is accessed.
    It is thread-safe and subprocess-safe, and is useful to measure how many times a sample is accessed.
    Because of atomic file writes, it is slow and should not be used in any performance measurements.
    """

    def __init__(self, *, total_dataset_size: int, data_shape: Sequence[int], memory_format: MemoryFormat, device: str,
                 num_classes: int, is_train: bool, tmpdir: str):
        super().__init__(total_dataset_size=total_dataset_size,
                         data_shape=data_shape,
                         num_classes=num_classes,
                         memory_format=memory_format,
                         device=device)
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
                total_dataset_size=self.total_dataset_size,
                data_shape=self.data_shape,
                num_classes=self.num_classes,
                device=self.device,
                memory_format=self.memory_format,
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


@pytest.mark.timeout(90)
@pytest.mark.parametrize("is_gpu,num_procs", [
    pytest.param(False, 1, id="1-cpu"),
    pytest.param(False, 2, id="2-cpu"),
    pytest.param(True, 1, marks=[pytest.mark.n_gpus(1)], id="1-gpu"),
    pytest.param(True, 2, marks=[pytest.mark.n_gpus(2)], id="2-gpu"),
])
def test_ddp(is_gpu: bool, num_procs: int, *, tmpdir: pathlib.Path, mosaic_trainer_hparams: TrainerHparams) -> None:
    with_distributed(num_procs, _test_ddp)(is_gpu, num_procs, tmpdir, mosaic_trainer_hparams)


def _test_ddp(is_gpu: bool, num_procs: int, tmpdir: pathlib.Path, mosaic_trainer_hparams: TrainerHparams) -> None:
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
    model_hparams = hparams.model
    assert isinstance(model_hparams, SimpleBatchPairModelHparams)
    model = model_hparams.initialize_object()
    shape = list(model.in_shape)  # type: ignore

    callback_registry["checkbatch0"] = CheckBatch0Hparams
    dataset_registry["tracked"] = TrackedDatasetHparams

    hparams.train_dataset = TrackedDatasetHparams(
        total_dataset_size=300,
        data_shape=shape,
        num_classes=model.num_classes,
        device="cpu",
        is_train=True,
        memory_format=MemoryFormat.CONTIGUOUS_FORMAT,
        tmpdir=str(tmpdir),
    )
    hparams.val_dataset = TrackedDatasetHparams(
        total_dataset_size=300,
        data_shape=shape,
        num_classes=model.num_classes,
        device="cpu",
        is_train=False,
        memory_format=MemoryFormat.CONTIGUOUS_FORMAT,
        tmpdir=str(tmpdir),
    )
    if is_gpu:
        device = GPUDeviceHparams()
    else:
        device = CPUDeviceHparams()
    hparams.device = device
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
    hparams.callbacks.append(CheckBatch0Hparams(tmpdir=tmpdir))
    trainer = hparams.initialize_object()
    assert trainer.state.world_size == num_procs
    assert trainer.state.local_world_size == num_procs
    assert isinstance(trainer.train_dl_spec.dataset, collections.abc.Sized)
    num_train_samples = len(trainer.train_dl_spec.dataset)
    assert isinstance(trainer.eval_dl_spec.dataset, collections.abc.Sized)
    num_eval_samples = len(trainer.eval_dl_spec.dataset)
    trainer.fit()

    # now validate that each sample were accessed exactly hparams.max_epochs * batch size times
    num_epochs = hparams.max_epochs

    for i in range(num_train_samples):
        for epoch in range(num_epochs):
            assert os.path.exists(get_file_path(
                tmpdir, idx=i, epoch=epoch, is_train=True)), f"train sample {i} was not accessed during epoch {epoch}"
        assert not os.path.exists(get_file_path(tmpdir, idx=i, epoch=num_epochs,
                                                is_train=True)), f"train sample {i} was accessed too many times"

    for i in range(num_eval_samples):
        for epoch in range(num_epochs):
            assert os.path.exists(get_file_path(
                tmpdir, idx=i, epoch=epoch, is_train=False)), f"val sample {i} was not accessed during epoch {epoch}"
        # the eval dataloader is spun once more to initialize the rng, so expecting num_epochs + 1 to not exist
        assert not os.path.exists(get_file_path(tmpdir, idx=i, epoch=num_epochs + 1,
                                                is_train=False)), f"val sample {i} was accessed too many times"

    is_train_to_pickles: Dict[bool, List[Dict[str, types.Tensor]]] = {True: [], False: []}

    for epoch in range(num_epochs):
        for local_rank in range(trainer.state.local_world_size):
            for is_train in (True, False):
                data: Dict[str, types.Tensor] = torch.load(  # type: ignore
                    get_batch_file_path(tmpdir, rank=local_rank, epoch=epoch, is_train=is_train),
                    map_location='cpu',
                )
                for pickle in is_train_to_pickles[is_train]:
                    assert not torch.all(data['last_input'] == pickle['last_input'])
                    assert not torch.all(data['last_target'] == pickle['last_target'])
                is_train_to_pickles[is_train].append(data)
