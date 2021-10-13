# Copyright 2021 MosaicML. All Rights Reserved.

import torch
from torchvision import datasets, transforms

import composer.models
from composer import DataloaderSpec, Trainer
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


# Your custom train DataloaderSpec
train_dataloader_spec = DataloaderSpec(
    dataset=datasets.MNIST('/datasets/', train=True, transform=transforms.ToTensor(), download=True),
    drop_last=False,
    shuffle=True,
)

# Your custom eval dataset
eval_dataloader_spec = DataloaderSpec(
    dataset=datasets.MNIST('/datasets/', train=False, transform=transforms.ToTensor()),
    drop_last=False,
    shuffle=False,
)

# Initialize Trainer with custom model, custom train and eval datasets, and algorithms to train with
trainer = Trainer(model=SimpleModel(num_hidden=128, num_classes=10),
                  train_dataloader_spec=train_dataloader_spec,
                  eval_dataloader_spec=eval_dataloader_spec,
                  max_epochs=3,
                  train_batch_size=256,
                  eval_batch_size=256,
                  algorithms=[CutOut(n_holes=1, length=10), LabelSmoothing(alpha=0.1)])

trainer.fit()
