# Copyright 2021 MosaicML. All Rights Reserved.

from copy import deepcopy

import torch

from composer.core.types import DataLoader
from composer.datasets.synthetic import SyntheticDataLabelType, SyntheticDatasetHparams, SyntheticDataType
from composer.models.base import BaseMosaicModel
from composer.models.classify_mnist.mnist_hparams import MnistClassifierHparams
from composer.optim.optimizer_hparams import SGDHparams
from composer.trainer.devices.device_gpu import DeviceGPU
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import ddp, ensure_tuple


def get_total_loss(model: BaseMosaicModel, dataloader: DataLoader):
    with torch.no_grad():
        total_loss = 0
        for batch in dataloader:
            outputs = model(batch)
            loss = model.loss(outputs, batch=batch)
            for l in ensure_tuple(loss):
                total_loss += l.item()

        total_loss_tensor = torch.Tensor([total_loss])
        ddp.all_reduce(total_loss_tensor)
        return total_loss_tensor.item() / ddp.get_world_size()


def train_model(mosaic_trainer_hparams: TrainerHparams, max_epochs: int = 2, run_loss_check: bool = False):
    total_dataset_size = 16
    mosaic_trainer_hparams.train_dataset = SyntheticDatasetHparams(total_dataset_size=total_dataset_size,
                                                                   data_shape=[1, 28, 28],
                                                                   data_type=SyntheticDataType.SEPARABLE,
                                                                   label_type=SyntheticDataLabelType.CLASSIFICATION_INT,
                                                                   num_classes=2,
                                                                   device="cpu",
                                                                   drop_last=True,
                                                                   shuffle=False)
    # Not used in the training loop only being set because it is required
    mosaic_trainer_hparams.val_dataset = SyntheticDatasetHparams(total_dataset_size=total_dataset_size,
                                                                 data_shape=[1, 28, 28],
                                                                 data_type=SyntheticDataType.SEPARABLE,
                                                                 label_type=SyntheticDataLabelType.CLASSIFICATION_INT,
                                                                 num_classes=2,
                                                                 device="cpu",
                                                                 drop_last=True,
                                                                 shuffle=False)

    mosaic_trainer_hparams.model = MnistClassifierHparams(num_classes=2)
    mosaic_trainer_hparams.optimizer = SGDHparams(lr=1e-2)
    mosaic_trainer_hparams.total_batch_size = total_dataset_size  # one batch per epoch
    mosaic_trainer_hparams.max_epochs = max_epochs
    # Don't validate
    mosaic_trainer_hparams.validate_every_n_epochs = max_epochs + 1

    trainer = Trainer.create_from_hparams(mosaic_trainer_hparams)

    original_model = deepcopy(trainer.state.model)
    assert isinstance(original_model, BaseMosaicModel)

    trainer.fit()

    # The original model is on the CPU so move it to GPU if needed
    if isinstance(trainer.device, DeviceGPU):
        original_model = trainer.device.module_to_device(original_model)

    if run_loss_check and trainer.state.train_dataloader:
        initial_loss = get_total_loss(original_model, trainer.state.train_dataloader)

        unwrapped_model = trainer.state.model.module
        assert isinstance(unwrapped_model, BaseMosaicModel)
        post_fit_loss = get_total_loss(unwrapped_model, trainer.state.train_dataloader)

        assert post_fit_loss < initial_loss + 1e-5
