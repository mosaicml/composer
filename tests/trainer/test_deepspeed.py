import collections.abc
import os
import pathlib
from typing import Dict, List

import pytest
import torch

import composer.core.types as types
from composer.datasets import DataloaderHparams, MemoryFormat
from composer.trainer.devices.device_hparams import GPUDeviceHparams
from composer.trainer.trainer_hparams import TrainerHparams, callback_registry, dataset_registry
from composer.utils import ddp
from tests.fixtures.models import SimpleBatchPairModelHparams
from tests.trainer.test_ddp import CheckBatch0Hparams, TrackedDatasetHparams, get_batch_file_path, get_file_path


@pytest.mark.timeout(90)
@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [pytest.param(1), pytest.param(2, marks=pytest.mark.world_size(2))])
def test_deepspeed(world_size: int, mosaic_trainer_hparams: TrainerHparams, tmpdir: pathlib.Path) -> None:
    """Pretty much just copied from ./test_ddp"""

    del world_size  # unused. Set via env variables

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
    hparams.device = GPUDeviceHparams()
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
    hparams.callbacks.append(CheckBatch0Hparams(tmpdir=str(tmpdir)))
    trainer = hparams.initialize_object()
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
        for local_rank in range(ddp.get_local_world_size()):
            for is_train in (True, False):
                data: Dict[str, types.Tensor] = torch.load(  # type: ignore
                    get_batch_file_path(tmpdir, rank=local_rank, epoch=epoch, is_train=is_train),
                    map_location='cpu',
                )
                for pickle in is_train_to_pickles[is_train]:
                    assert not torch.all(data['last_input'] == pickle['last_input'])
                    assert not torch.all(data['last_target'] == pickle['last_target'])
                is_train_to_pickles[is_train].append(data)
