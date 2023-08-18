# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Example script to train a ResNet model on ImageNet."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer import Trainer
from composer.algorithms import ChannelsLast, CutMix, LabelSmoothing
from composer.loggers import NeptuneLogger
from composer.models import mnist_model

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('data', download=True, train=True, transform=transform)
eval_dataset = datasets.MNIST('data', download=True, train=False, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128)
eval_dataloader = DataLoader(eval_dataset, batch_size=128)
logger = NeptuneLogger()

trainer = Trainer(
    model=mnist_model(),
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep',
    algorithms=[
        ChannelsLast(),
        CutMix(alpha=1.0),
        LabelSmoothing(smoothing=0.1),
    ],
    loggers=logger,
)
trainer.fit()
