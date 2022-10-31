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
class LenetModel(composer.models.ComposerClassifier):
    """Your custom model."""

    def __init__(self, num_hidden: int, num_classes: int) -> None:
        module = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(18, 48, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            
            #nn.Conv2d(16, 120, kernel_size=(5,5)),
            #nn.ReLU(inplace=True),
            nn.Flatten(1),
            
            nn.Linear(16*5*5*3, 360),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(360, 84*3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(84*3, num_classes),
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

model=LenetModel(num_hidden=64, num_classes=10).to("cuda")

optimizer = composer.optim.DecoupledSGDW(
    model.parameters(),
    lr = 0.01,
    momentum = 0.9,
    weight_decay=0.0005,
    #l2normalization -update 후 loss_function
    #l2-normalization, l1-normalization
    #weight값을 평준화 l2-normalization(weight decay) (l2-reguralization)
    #adamw- weight 에서만 l2를 주는 것
)
scheduler = composer.optim.MultiStepScheduler(
    milestones= '20ep',
    gamma=0.1,
)

#criterion = composer.optim.cross
# Initialize Trainer with custom model, custom train and eval datasets, and algorithms to train with
trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='80ep',
                  optimizers=optimizer,
                  algorithms=[GyroDropout(1024,8)]
                )

trainer.fit()
