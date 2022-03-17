# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Tuple, Type, Union
from unittest.mock import Mock

import pytest
import torch
import torch.utils.data
from torch.optim import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy

from composer.core import DataSpec, Evaluator, Precision, State
from composer.core.types import DataLoader, PyTorchScheduler
from composer.datasets import DataLoaderHparams, DatasetHparams
from composer.loggers import Logger
from composer.models import ComposerClassifier, ModelHparams
from composer.optim import AdamHparams, ExponentialSchedulerHparams
from composer.trainer import TrainerHparams
from composer.trainer.devices import CPUDeviceHparams
from tests.fixtures.models import (SimpleBatchPairModel, SimpleConvModel, _SimpleBatchPairModelHparams,
                                   _SimpleDatasetHparams, _SimplePILDatasetHparams)


@pytest.fixture
def dummy_in_shape() -> Tuple[int, ...]:
    return (1, 5, 5)


@pytest.fixture
def dummy_num_classes() -> int:
    return 3


@pytest.fixture()
def dummy_train_batch_size() -> int:
    return 16


@pytest.fixture()
def dummy_val_batch_size() -> int:
    return 32


@pytest.fixture
def dummy_model_hparams(
        dummy_in_shape: Tuple[int, ...], dummy_num_classes: int,
        SimpleBatchPairModelHparams: Type[_SimpleBatchPairModelHparams]) -> _SimpleBatchPairModelHparams:
    return SimpleBatchPairModelHparams(num_channels=dummy_in_shape[0], num_classes=dummy_num_classes)


@pytest.fixture
def dummy_model(dummy_model_hparams: _SimpleBatchPairModelHparams) -> SimpleBatchPairModel:
    return dummy_model_hparams.initialize_object()


@pytest.fixture
def dummy_train_dataset_hparams(dummy_model: SimpleBatchPairModel, dummy_in_shape: Tuple[int],
                                SimpleDatasetHparams: Type[_SimpleDatasetHparams]) -> DatasetHparams:
    return SimpleDatasetHparams(
        use_synthetic=True,
        drop_last=True,
        shuffle=False,
        num_classes=dummy_model.num_classes,
        data_shape=list(dummy_in_shape),
    )


@pytest.fixture
def dummy_train_pil_dataset_hparams(dummy_model: SimpleBatchPairModel, dummy_in_shape: Tuple[int],
                                    SimplePILDatasetHparams: Type[_SimplePILDatasetHparams]) -> DatasetHparams:
    return SimplePILDatasetHparams(
        use_synthetic=True,
        drop_last=True,
        shuffle=False,
        num_classes=dummy_model.num_classes,
        data_shape=list(dummy_in_shape)[1:],
    )


@pytest.fixture
def dummy_val_dataset_hparams(dummy_model: SimpleBatchPairModel, dummy_in_shape: Tuple[int],
                              SimpleDatasetHparams: Type[_SimpleDatasetHparams]) -> DatasetHparams:
    return SimpleDatasetHparams(
        use_synthetic=True,
        drop_last=False,
        shuffle=False,
        num_classes=dummy_model.num_classes,
        data_shape=list(dummy_in_shape),
    )


@pytest.fixture
def dummy_optimizer(dummy_model: SimpleBatchPairModel):
    return torch.optim.SGD(dummy_model.parameters(), lr=0.001)


@pytest.fixture
def dummy_scheduler(dummy_optimizer: Optimizer):
    return torch.optim.lr_scheduler.LambdaLR(dummy_optimizer, lambda _: 1.0)


@pytest.fixture()
def dummy_state(dummy_model: SimpleBatchPairModel, dummy_train_dataloader: DataLoader, dummy_optimizer: Optimizer,
                dummy_scheduler: PyTorchScheduler, dummy_val_dataloader: DataLoader, rank_zero_seed: int) -> State:
    evaluators = [
        Evaluator(label="dummy_label", dataloader=dummy_val_dataloader, metrics=dummy_model.metrics(train=False))
    ]
    state = State(
        model=dummy_model,
        precision=Precision.FP32,
        grad_accum=1,
        rank_zero_seed=rank_zero_seed,
        train_dataloader=dummy_train_dataloader,
        evaluators=evaluators,
        optimizers=dummy_optimizer,
        max_duration="10ep",
    )
    state.schedulers = dummy_scheduler

    return state


@pytest.fixture
def dummy_dataloader_hparams() -> DataLoaderHparams:
    return DataLoaderHparams(
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        timeout=0.0,
    )


@pytest.fixture
def dummy_train_dataloader(dummy_train_dataset_hparams: DatasetHparams, dummy_train_batch_size: int,
                           dummy_dataloader_hparams: DataLoaderHparams) -> Union[DataLoader, DataSpec]:
    return dummy_train_dataset_hparams.initialize_object(dummy_train_batch_size, dummy_dataloader_hparams)


@pytest.fixture
def dummy_train_pil_dataloader(dummy_train_pil_dataset_hparams: DatasetHparams, dummy_train_batch_size: int,
                               dummy_dataloader_hparams: DataLoaderHparams) -> Union[DataLoader, DataSpec]:
    return dummy_train_pil_dataset_hparams.initialize_object(dummy_train_batch_size, dummy_dataloader_hparams)


@pytest.fixture
def dummy_val_dataloader(dummy_train_dataset_hparams: DatasetHparams, dummy_val_batch_size: int,
                         dummy_dataloader_hparams: DataLoaderHparams) -> Union[DataLoader, DataSpec]:
    return dummy_train_dataset_hparams.initialize_object(dummy_val_batch_size, dummy_dataloader_hparams)


@pytest.fixture()
def dummy_logger(dummy_state: State):
    return Logger(dummy_state)


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
def composer_trainer_hparams(
    dummy_model_hparams: ModelHparams,
    dummy_train_dataset_hparams: DatasetHparams,
    dummy_val_dataset_hparams: DatasetHparams,
    dummy_train_batch_size: int,
    dummy_val_batch_size: int,
    rank_zero_seed: int,
) -> TrainerHparams:
    return TrainerHparams(
        algorithms=[],
        optimizer=AdamHparams(),
        schedulers=[ExponentialSchedulerHparams(gamma=0.1)],
        max_duration="2ep",
        precision=Precision.FP32,
        train_batch_size=dummy_train_batch_size,
        eval_batch_size=dummy_val_batch_size,
        seed=rank_zero_seed,
        dataloader=DataLoaderHparams(
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=False,
            pin_memory=False,
            timeout=0.0,
        ),
        device=CPUDeviceHparams(),
        deterministic_mode=True,
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
def state_with_model(simple_conv_model: torch.nn.Module, dummy_train_dataloader: DataLoader,
                     dummy_val_dataloader: DataLoader, rank_zero_seed: int):
    metric_coll = MetricCollection([Accuracy()])
    evaluators = [Evaluator(label="dummy_label", dataloader=dummy_val_dataloader, metrics=metric_coll)]
    state = State(
        grad_accum=1,
        rank_zero_seed=rank_zero_seed,
        max_duration="100ep",
        model=simple_conv_model,
        precision=Precision.FP32,
        train_dataloader=dummy_train_dataloader,
        evaluators=evaluators,
    )
    return state


@pytest.fixture()
def simple_conv_model():
    return ComposerClassifier(SimpleConvModel())


@pytest.fixture(scope="session")
def SimpleBatchPairModelHparams():
    TrainerHparams.register_class("model", _SimpleBatchPairModelHparams, "simple_batch_pair_model")
    return _SimpleBatchPairModelHparams


@pytest.fixture(scope="session")
def SimpleDatasetHparams():
    TrainerHparams.register_class("train_dataset", _SimpleDatasetHparams, "simple_dataset")
    return _SimpleDatasetHparams


@pytest.fixture(scope="session")
def SimplePILDatasetHparams():
    TrainerHparams.register_class("train_dataset", _SimplePILDatasetHparams, "simple_pil_dataset")
    return _SimplePILDatasetHparams
