# Copyright 2021 MosaicML. All Rights Reserved.

import torch
import torch.utils.data
from torchvision import datasets, transforms

import composer.models
from composer import Trainer
# Example algorithms to train with
from composer.algorithms import CutOut, LabelSmoothing


# Your custom model
class SimpleModel(composer.models.MosaicClassifier):

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
                  train_data=train_dataloader,
                  eval_data=eval_dataloader,
                  max_epochs=3,
                  algorithms=[CutOut(n_holes=1, length=10), LabelSmoothing(alpha=0.1)])

trainer.fit()
