# Copyright 2021 MosaicML. All Rights Reserved.

import itertools
from copy import deepcopy

import pytest
import torch

from composer.core.types import DataLoader
from composer.datasets.mnist import MNISTDatasetHparams
from composer.models.base import ComposerModel
from composer.models.classify_mnist.mnist_hparams import MnistClassifierHparams
from composer.optim.optimizer_hparams import SGDHparams
from composer.trainer.devices.device import Device
from composer.trainer.devices.device_gpu import DeviceGPU
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import dist, ensure_tuple


def get_total_loss(model: ComposerModel, dataloader: DataLoader, device: Device):
    with torch.no_grad():
        total_loss = 0
        for batch in itertools.islice(dataloader, 1):
            batch = device.batch_to_device(batch)
            outputs = model(batch)
            loss = model.loss(outputs, batch=batch)
            for l in ensure_tuple(loss):
                total_loss += l.item()

        total_loss_tensor = device.tensor_to_device(torch.Tensor([total_loss]))
        dist.all_reduce(total_loss_tensor)
        return total_loss_tensor.item() / dist.get_world_size()


def train_model(composer_trainer_hparams: TrainerHparams, max_epochs: int = 2, run_loss_check: bool = False):
    pytest.xfail("train_model is flaky")
    total_dataset_size = 16
    composer_trainer_hparams.train_dataset = MNISTDatasetHparams(use_synthetic=True,)
    composer_trainer_hparams.train_subset_num_batches = 1
    composer_trainer_hparams.val_dataset = MNISTDatasetHparams(use_synthetic=True,)
    composer_trainer_hparams.eval_subset_num_batches = 1
    composer_trainer_hparams.model = MnistClassifierHparams(num_classes=10)
    composer_trainer_hparams.optimizer = SGDHparams(lr=1e-2)
    composer_trainer_hparams.train_batch_size = total_dataset_size  # one batch per epoch
    composer_trainer_hparams.max_duration = f"{max_epochs}ep"
    # Don't validate
    composer_trainer_hparams.validate_every_n_epochs = max_epochs + 1

    trainer = composer_trainer_hparams.initialize_object()

    original_model = deepcopy(trainer.state.model)
    assert isinstance(original_model, ComposerModel)

    trainer.fit()

    # The original model is on the CPU so move it to GPU if needed
    if isinstance(trainer._device, DeviceGPU):
        original_model = trainer._device.module_to_device(original_model)

    if run_loss_check and trainer.state.train_dataloader:
        initial_loss = get_total_loss(original_model, trainer.state.train_dataloader, trainer._device)

        unwrapped_model = trainer.state.model.module
        assert isinstance(unwrapped_model, ComposerModel)
        post_fit_loss = get_total_loss(unwrapped_model, trainer.state.train_dataloader, trainer._device)
        assert post_fit_loss < initial_loss + 1e-5, f"post_fit_loss({post_fit_loss}) - initial_loss({initial_loss}) >= 1e-5"
