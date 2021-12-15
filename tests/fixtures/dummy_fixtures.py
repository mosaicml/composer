# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Tuple, Type, Union
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.utils.data

from composer import Logger, State
from composer.core.types import DataLoader, Model, Precision
from composer.datasets import DataloaderHparams, DatasetHparams
from composer.datasets.hparams import DataloaderSpec
from composer.models import ModelHparams, MosaicClassifier
from composer.optim import AdamHparams, ExponentialLRHparams
from composer.trainer import TrainerHparams
from composer.trainer.devices import CPUDeviceHparams
from tests.fixtures.models import (SimpleBatchPairModel, SimpleConvModel, _SimpleBatchPairModelHparams,
                                   _SimpleDatasetHparams)


@pytest.fixture
def dummy_in_shape() -> Tuple[int, ...]:
    return (1, 5, 5)


@pytest.fixture
def dummy_num_classes() -> int:
    return 3


@pytest.fixture()
def dummy_train_batch_size() -> int:
    return 64


@pytest.fixture()
def dummy_val_batch_size() -> int:
    return 128


@pytest.fixture
def dummy_model_hparams(
        dummy_in_shape: Tuple[int, ...], dummy_num_classes: int,
        SimpleBatchPairModelHparams: Type[_SimpleBatchPairModelHparams]) -> _SimpleBatchPairModelHparams:
    return SimpleBatchPairModelHparams(in_shape=list(dummy_in_shape), num_classes=dummy_num_classes)


@pytest.fixture
def dummy_model(dummy_model_hparams: _SimpleBatchPairModelHparams) -> SimpleBatchPairModel:
    return dummy_model_hparams.initialize_object()


@pytest.fixture
def dummy_train_dataset_hparams(dummy_model: SimpleBatchPairModel,
                                SimpleDatasetHparams: Type[_SimpleDatasetHparams]) -> DatasetHparams:
    return SimpleDatasetHparams(
        use_synthetic=True,
        drop_last=True,
        shuffle=False,
        num_classes=dummy_model.num_classes,
        data_shape=list(dummy_model.in_shape),
    )


@pytest.fixture
def dummy_val_dataset_hparams(dummy_model: SimpleBatchPairModel,
                              SimpleDatasetHparams: Type[_SimpleDatasetHparams]) -> DatasetHparams:
    return SimpleDatasetHparams(
        use_synthetic=True,
        drop_last=False,
        shuffle=False,
        num_classes=dummy_model.num_classes,
        data_shape=list(dummy_model.in_shape),
    )


@pytest.fixture()
def dummy_state_without_rank(dummy_model: SimpleBatchPairModel, dummy_train_dataloader: DataLoader,
                             dummy_val_dataloader: DataLoader) -> State:
    state = State(
        model=dummy_model,
        precision=Precision.FP32,
        grad_accum=1,
        train_dataloader=dummy_train_dataloader,
        eval_dataloader=dummy_val_dataloader,
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
        timeout=0.0,
    )


@pytest.fixture
def dummy_train_dataloader(dummy_train_dataset_hparams: DatasetHparams, dummy_train_batch_size: int,
                           dummy_dataloader_hparams: DataloaderHparams) -> Union[DataLoader, DataloaderSpec]:
    return dummy_train_dataset_hparams.initialize_object(dummy_train_batch_size, dummy_dataloader_hparams)


@pytest.fixture
def dummy_val_dataloader(dummy_train_dataset_hparams: DatasetHparams, dummy_val_batch_size: int,
                         dummy_dataloader_hparams: DataloaderHparams) -> Union[DataLoader, DataloaderSpec]:
    return dummy_train_dataset_hparams.initialize_object(dummy_val_batch_size, dummy_dataloader_hparams)


@pytest.fixture()
def dummy_state(dummy_state_without_rank: State) -> State:
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
    dummy_train_dataset_hparams: DatasetHparams,
    dummy_val_dataset_hparams: DatasetHparams,
    dummy_train_batch_size: int,
    dummy_val_batch_size: int,
) -> TrainerHparams:
    return TrainerHparams(
        algorithms=[],
        optimizer=AdamHparams(),
        schedulers=[ExponentialLRHparams(gamma=0.1)],
        max_epochs=2,
        precision=Precision.FP32,
        train_batch_size=dummy_train_batch_size,
        eval_batch_size=dummy_val_batch_size,
        dataloader=DataloaderHparams(
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=False,
            pin_memory=False,
            timeout=0.0,
        ),
        device=CPUDeviceHparams(),
        loggers=[],
        model=dummy_model_hparams,
        val_dataset=dummy_val_dataset_hparams,
        train_dataset=dummy_train_dataset_hparams,
        grad_accum=1,
        train_subset_num_batches=3,
        eval_subset_num_batches=3,
    )


@pytest.fixture()
def simple_conv_model_input():
    return torch.rand((64, 32, 64, 64))


@pytest.fixture()
def state_with_model(simple_conv_model: Model, dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader):
    state = State(
        grad_accum=1,
        max_epochs=100,
        model=simple_conv_model,
        precision=Precision.FP32,
        train_dataloader=dummy_train_dataloader,
        eval_dataloader=dummy_val_dataloader,
    )
    return state


@pytest.fixture()
def simple_conv_model():
    return MosaicClassifier(SimpleConvModel())


@pytest.fixture(scope="session")
def SimpleBatchPairModelHparams():
    TrainerHparams.register_class("model", _SimpleBatchPairModelHparams, "simple_batch_pair_model")
    return _SimpleBatchPairModelHparams


@pytest.fixture(scope="session")
def SimpleDatasetHparams():
    TrainerHparams.register_class("train_dataset", _SimpleDatasetHparams, "simple_dataset")
    return _SimpleDatasetHparams
