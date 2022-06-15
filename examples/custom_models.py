# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Example for training with an algorithm on a custom model."""

import torch
import torch.utils.data
from torchvision import datasets, transforms

import composer.models
from composer import Trainer
# Example algorithms to train with
from composer.algorithms import CutOut, LabelSmoothing


# Your custom model
class SimpleModel(composer.models.ComposerClassifier):
    """Your custom model."""

    def __init__(self, num_hidden: int, num_classes: int):
        module = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(28 * 28, num_hidden),
            torch.nn.Linear(num_hidden, num_classes),
        )
        self.num_classes = num_classes
        super().__init__(module=module)


# Your custom train dataloader
train_dataloader = torch.utils.data.DataLoader(
    dataset=datasets.MNIST('/datasets/', train=True, transform=transforms.ToTensor(), download=True),
    drop_last=False,
    shuffle=True,
    batch_size=256,
)

# Your custom eval dataloader
eval_dataloader = torch.utils.data.DataLoader(
    dataset=datasets.MNIST('/datasets/', train=False, transform=transforms.ToTensor()),
    drop_last=False,
    shuffle=False,
    batch_size=256,
)

# Initialize Trainer with custom model, custom train and eval datasets, and algorithms to train with
trainer = Trainer(model=SimpleModel(num_hidden=128, num_classes=10),
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='3ep',
                  algorithms=[CutOut(num_holes=1, length=0.5), LabelSmoothing(0.1)])

trainer.fit()
