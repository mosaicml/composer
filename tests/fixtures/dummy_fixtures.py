# Copyright 2021 MosaicML. All Rights Reserved.

import os
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.distributed as dist
import torch.utils.data
from _pytest.monkeypatch import MonkeyPatch

from composer import Logger, State
from composer.core.types import DataLoader, Model, Precision, Tensors
from composer.datasets import DataloaderHparams, DataloaderSpec, DatasetHparams, SyntheticDatasetHparams
from composer.models import BaseMosaicModel, MnistClassifierHparams, ModelHparams, MosaicClassifier
from composer.optim import AdamHparams, ExponentialLRHparams
from composer.trainer import TrainerHparams
from composer.trainer.ddp import DDPDataLoader, DDPHparams, FileStoreHparams
from composer.trainer.devices import CPUDeviceHparams


@pytest.fixture()
def dummy_model_hparams() -> ModelHparams:
    return MnistClassifierHparams(num_classes=10)


@pytest.fixture()
def dummy_model(dummy_model_hparams: ModelHparams) -> BaseMosaicModel:
    return dummy_model_hparams.initialize_object()


@pytest.fixture()
def dummy_dataset_hparams() -> SyntheticDatasetHparams:
    return SyntheticDatasetHparams(num_classes=10,
                                   shape=[1, 28, 28],
                                   device="cpu",
                                   sample_pool_size=256,
                                   one_hot=False,
                                   drop_last=True,
                                   shuffle=False)


@pytest.fixture()
def dummy_dataloader_spec(dummy_dataset_hparams: SyntheticDatasetHparams) -> DataloaderSpec:
    return dummy_dataset_hparams.initialize_object()


@pytest.fixture()
def dummy_state_without_rank(dummy_model: BaseMosaicModel) -> State:
    state = State(
        model=dummy_model,
        epoch=5,
        step=50,
        precision=Precision.FP32,
        grad_accum=1,
        train_batch_size=10,
        eval_batch_size=10,
        max_epochs=10,
    )
    return state


@pytest.fixture
def dummy_dataloader_hparams() -> DataloaderHparams:
    return DataloaderHparams(
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        timeout=0,
    )


def get_dataloader(dataloader_spec: DataloaderSpec, dataloader_hparams: DataloaderHparams) -> DataLoader:
    batch_size = 10

    sampler = torch.utils.data.DistributedSampler[int](
        dataloader_spec.dataset,
        drop_last=dataloader_spec.drop_last,
        shuffle=dataloader_spec.shuffle,
        rank=0,
        num_replicas=1,
    )

    dataloader = torch.utils.data.DataLoader(
        dataloader_spec.dataset,
        batch_size=batch_size,
        shuffle=False,  # set in the sampler
        num_workers=dataloader_hparams.num_workers,
        pin_memory=dataloader_hparams.pin_memory,
        drop_last=dataloader_spec.drop_last,
        sampler=sampler,
        collate_fn=dataloader_spec.collate_fn,
        worker_init_fn=dataloader_spec.worker_init_fn,
        multiprocessing_context=dataloader_spec.multiprocessing_context,
        generator=dataloader_spec.generator,
        timeout=dataloader_hparams.timeout,
        prefetch_factor=dataloader_hparams.prefetch_factor,
        persistent_workers=dataloader_hparams.persistent_workers,
    )
    return DDPDataLoader(dataloader)


@pytest.fixture
def dummy_train_dataloader(dummy_dataloader_spec: DataloaderSpec,
                           dummy_dataloader_hparams: DataloaderHparams) -> DataLoader:
    return get_dataloader(dummy_dataloader_spec, dummy_dataloader_hparams)


@pytest.fixture
def dummy_val_dataloader(dummy_dataloader_spec: DataloaderSpec,
                         dummy_dataloader_hparams: DataloaderHparams) -> DataLoader:
    return get_dataloader(dummy_dataloader_spec, dummy_dataloader_hparams)


@pytest.fixture()
def dummy_state(dummy_state_without_rank: State, monkeypatch: MonkeyPatch) -> State:
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    return dummy_state_without_rank


@pytest.fixture()
def dummy_state_dl(dummy_state: State, dummy_train_dataloader: DataLoader) -> State:
    dummy_state.train_dataloader = dummy_train_dataloader
    return dummy_state


@pytest.fixture()
def dummy_logger(dummy_state: State):
    return Logger(dummy_state)


@pytest.fixture
def logger_mock():
    return MagicMock()


"""
Dummy algorithms
"""


@pytest.fixture()
def algorithms(always_match_algorithms):
    return always_match_algorithms


@pytest.fixture()
def always_match_algorithms():
    attrs = {'match.return_value': True}
    return [Mock(**attrs) for _ in range(5)]


@pytest.fixture()
def never_match_algorithms():
    attrs = {'match.return_value': False}
    return [Mock(**attrs) for _ in range(5)]


@pytest.fixture
def mosaic_trainer_hparams(
    dummy_model_hparams: ModelHparams,
    dummy_dataset_hparams: DatasetHparams,
    ddp_tmpdir: str,
) -> TrainerHparams:
    return TrainerHparams(
        algorithms=[],
        optimizer=AdamHparams(),
        schedulers=[ExponentialLRHparams(gamma=0.1)],
        max_epochs=2,
        precision=Precision.FP32,
        total_batch_size=64,
        eval_batch_size=64,
        ddp=DDPHparams(
            store=FileStoreHparams(os.path.join(ddp_tmpdir, "store")),
            node_rank=0,
            num_nodes=1,
            fork_rank_0=False,
        ),
        dataloader=DataloaderHparams(
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=False,
            pin_memory=False,
            timeout=0,
        ),
        device=CPUDeviceHparams(n_cpus=1),
        loggers=[],
        model=dummy_model_hparams,
        val_dataset=dummy_dataset_hparams,
        train_dataset=dummy_dataset_hparams,
        grad_accum=1,
    )


"""
Simple models
"""


class SimpleConvModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        conv_args = dict(kernel_size=(3, 3), padding=1)
        self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=8, stride=2, bias=False, **conv_args)  # stride > 1
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=32, stride=2, bias=False,
                                     **conv_args)  # stride > 1 but in_channels < 16
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, stride=1, bias=False, **conv_args)  # stride = 1

        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)

    def forward(self, x: Tensors) -> Tensors:  # type: ignore
        # Very basic forward operation with no activation functions
        # used just to test that model surgery doesn't create forward prop bugs.
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool1(out)
        return out


@pytest.fixture()
def simple_conv_model():
    return MosaicClassifier(SimpleConvModel())


@pytest.fixture()
def simple_conv_model_input():
    return torch.rand((64, 32, 64, 64))


@pytest.fixture()
def state_with_model(simple_conv_model: Model):
    state = State(
        epoch=50,
        step=50,
        train_batch_size=100,
        eval_batch_size=100,
        grad_accum=1,
        max_epochs=100,
        model=simple_conv_model,
        precision=Precision.FP32,
    )
    return state
