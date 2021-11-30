# Copyright 2021 MosaicML. All Rights Reserved.

import collections.abc
import os
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Union

import pytest
import torch
import torch.distributed
import yahp as hp
from _pytest.monkeypatch import MonkeyPatch

import composer.core.types as types
from composer import Callback, Event
from composer.callbacks import CallbackHparams
from composer.core.logging import Logger
from composer.core.state import State
from composer.datasets import DataloaderHparams, synthetic
from composer.models.model_hparams import ModelHparams
from composer.trainer.devices import CPUDeviceHparams, GPUDeviceHparams
from composer.trainer.devices.device_hparams import DeviceHparams
from composer.trainer.trainer_hparams import TrainerHparams, callback_registry, dataset_registry
from composer.utils import ddp


def get_file_path(tmpdir: Union[str, pathlib.Path], *, rank: int, is_train: bool) -> str:
    train_str = "train" if is_train else "val"
    return os.path.join(tmpdir, f"{train_str}_rank_{rank}_num_accesses")


def get_batch_file_path(tmpdir: Union[str, pathlib.Path], *, rank: int, epoch: int, is_train: bool) -> str:
    train_str = "train" if is_train else "val"
    return os.path.join(tmpdir, f"{train_str}-rank-{rank}-epoch-{epoch}-batch0.pt")


class TrackedDataset(synthetic.SyntheticBatchPairDataset):
    """
    TrackedDataset atomically writes a file every time a record is accessed.
    It is thread-safe and subprocess-safe, and is useful to measure how many times a sample is accessed.
    Because of atomic file writes, it is slow and should not be used in any performance measurements.
    """

    def __init__(self, is_train: bool, tmpdir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_train = is_train
        self.tmpdir = tmpdir
        self.counter = 0

    def __getitem__(self, idx: int):
        self.counter += 1
        keyword = "train" if self.is_train else "val"
        with open(get_file_path(self.tmpdir, rank=ddp.get_global_rank(), is_train=self.is_train), "w+") as f:
            f.write(str(self.counter))
        return super().__getitem__(idx)


@dataclass
class TrackedDatasetHparams(synthetic.SyntheticBatchPairDatasetHparams):
    is_train: Optional[bool] = hp.optional("is_train", default=None)
    tmpdir: Optional[str] = hp.optional("tmpdir", default=None)

    def initialize_object(self, *args, **kwargs) -> TrackedDataset:
        assert self.is_train is not None
        assert self.tmpdir is not None
        return TrackedDataset(is_train=self.is_train, tmpdir=self.tmpdir, *args, **kwargs)


class CheckBatch0(Callback):

    def __init__(self, tmpdir: str):
        super().__init__()
        self.tmpdir = tmpdir

    def _run_event(self, event: Event, state: State, logger: Logger) -> None:
        if event in (Event.BEFORE_FORWARD, Event.EVAL_BEFORE_FORWARD):
            filepath = get_batch_file_path(self.tmpdir,
                                           rank=ddp.get_global_rank(),
                                           epoch=state.epoch,
                                           is_train=state.model.training)
            if os.path.exists(filepath):
                return
            last_input, last_target = state.batch_pair
            torch.save(  # type: ignore
                {
                    "last_input": last_input,
                    "last_target": last_target,
                }, filepath)


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
@pytest.mark.parametrize("device", [
    pytest.param(CPUDeviceHparams(), id="cpu"),
    pytest.param(GPUDeviceHparams(), id="gpu", marks=pytest.mark.gpu),
])
@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
def test_ddp(device: DeviceHparams, world_size: int, ddp_tmpdir: str, mosaic_trainer_hparams: TrainerHparams,
             SimpleBatchPairModelHparams: Type[ModelHparams]) -> None:
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
    del world_size  # unused. Set via env variables

    hparams = mosaic_trainer_hparams
    model_hparams = hparams.model
    assert isinstance(model_hparams, SimpleBatchPairModelHparams)

    callback_registry["checkbatch0"] = CheckBatch0Hparams
    dataset_registry["tracked"] = TrackedDatasetHparams

    hparams.total_batch_size = 10
    train_num_total_batches = 3
    hparams.train_dataset.set_num_total_batches(train_num_total_batches)
    hparams.train_dataset.set_synthetic(
        TrackedDatasetHparams(
            num_unique_samples_to_create=hparams.total_batch_size * train_num_total_batches,
            device="cpu",
            is_train=True,
            memory_format=synthetic.MemoryFormat.CONTIGUOUS_FORMAT,
            tmpdir=ddp_tmpdir,
        ))
    val_num_total_batches = 3
    hparams.eval_batch_size = 10
    hparams.val_dataset.set_num_total_batches(val_num_total_batches)
    hparams.val_dataset.set_synthetic(
        TrackedDatasetHparams(
            num_unique_samples_to_create=hparams.eval_batch_size * val_num_total_batches,
            device="cpu",
            is_train=False,
            memory_format=synthetic.MemoryFormat.CONTIGUOUS_FORMAT,
            tmpdir=ddp_tmpdir,
        ))
    hparams.val_dataset.shuffle = True
    hparams.device = device
    hparams.dataloader = DataloaderHparams(
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        timeout=0,
    )
    hparams.max_epochs = 2
    hparams.precision = types.Precision.FP32
    hparams.loggers = []
    hparams.validate_every_n_batches = 0
    hparams.validate_every_n_epochs = 1
    hparams.callbacks.append(CheckBatch0Hparams(tmpdir=ddp_tmpdir))
    trainer = hparams.initialize_object()
    assert isinstance(trainer.state.train_dataloader.dataset, collections.abc.Sized)
    num_train_samples = len(trainer.state.train_dataloader.dataset)
    assert isinstance(trainer.state.eval_dataloader.dataset, collections.abc.Sized)
    num_eval_samples = len(trainer.state.eval_dataloader.dataset)
    trainer.fit()

    expected_train_num_loads = hparams.max_epochs * hparams.total_batch_size * train_num_total_batches
    expected_val_num_loads = hparams.max_epochs * hparams.eval_batch_size * val_num_total_batches
    # adding hparams.eval_batch_size to account for the extra spin of the eval dataloader
    # that is called to create a deterministic ordering for the sampler
    expected_val_num_loads += hparams.eval_batch_size

    actual_train_num_loads = 0
    actual_val_num_loads = 0

    for i in range(ddp.get_world_size()):
        with open(get_file_path(ddp_tmpdir, is_train=True, rank=i), "r") as f:
            actual_train_num_loads += int(f.read())
        with open(get_file_path(ddp_tmpdir, is_train=False, rank=i), "r") as f:
            actual_val_num_loads += int(f.read())
    assert actual_train_num_loads == expected_train_num_loads, f"actual_train_num_loads({actual_train_num_loads}) != expected_train_num_loads({expected_train_num_loads})"
    assert actual_val_num_loads == expected_val_num_loads, f"actual_val_num_loads({actual_val_num_loads}) != expected_val_num_loads({expected_val_num_loads})"

    is_train_to_pickles: Dict[bool, List[Dict[str, types.Tensor]]] = {True: [], False: []}

    for epoch in range(hparams.max_epochs):
        for local_rank in range(ddp.get_local_world_size()):
            for is_train in (True, False):
                data: Dict[str, types.Tensor] = torch.load(  # type: ignore
                    get_batch_file_path(ddp_tmpdir, rank=local_rank, epoch=epoch, is_train=is_train),
                    map_location='cpu',
                )
                for pickle in is_train_to_pickles[is_train]:
                    assert not torch.all(
                        data['last_input'] == pickle['last_input']
                    ), f"inputs are the same for is_train={is_train}, epoch={epoch}, local_rank={local_rank}"
                    assert not torch.all(
                        data['last_target'] == pickle['last_target']
                    ), f"targets are the same for is_train={is_train}, epoch={epoch}, local_rank={local_rank}"
                is_train_to_pickles[is_train].append(data)
