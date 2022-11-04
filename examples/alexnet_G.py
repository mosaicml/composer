# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Example for training with an algorithm on a custom model."""

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms

import composer.models
import composer.optim
from composer import Trainer
# Example algorithms to train with
from composer.algorithms import GyroDropout

# Your custom model
class AlexnetModel(composer.models.ComposerClassifier):
    """Your custom model."""
    print("G1")
    def __init__(self, num_hidden: int, num_classes: int) -> None:
        module = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.AdaptiveAvgPool2d((6, 6)),
            
            nn.Flatten(1),       
    
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )


        super().__init__(module=module)


# Your custom train dataloader
train_dataloader = torch.utils.data.DataLoader(
    dataset=datasets.CIFAR10('/datasets/', train=True, transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]), download=True),
    drop_last=False,
    shuffle=True,
    batch_size=256,
)
# Your custom eval dataloader
eval_dataloader = torch.utils.data.DataLoader(
    dataset=datasets.CIFAR10('/datasets/', train=False, transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])),
    drop_last=False,
    shuffle=False,
    batch_size=256,
)

model=AlexnetModel(num_hidden=64, num_classes=10).to("cuda")

optimizer = composer.optim.DecoupledSGDW(
    model.parameters(),
    lr = 0.05,
    momentum = 0.9,
    weight_decay=0.0005,
)
scheduler = composer.optim.MultiStepScheduler(
    milestones= '40ep',
    gamma=0.1,
)

# Initialize Trainer with custom model, custom train and eval datasets, and algorithms to train with
trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='100ep',
                  optimizers=optimizer,
                  algorithms=[GyroDropout(0.5,256,16,196, 100)],
                )

trainer.fit()
