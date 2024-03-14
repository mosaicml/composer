# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Profiling Example.

For a walk-through of this example, please see the `profiling guide</trainer/performance_tutorials/profiling>`_.
"""

# [imports-start]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer import Trainer
from composer.models.tasks import ComposerClassifier
from composer.profiler import JSONTraceHandler, cyclic_schedule
from composer.profiler.profiler import Profiler

# [imports-end]

# [dataloader-start]
# Specify Dataset and Instantiate DataLoader
batch_size = 2048
data_directory = '~/datasets'

mnist_transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(data_directory, train=True, download=True, transform=mnist_transforms)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
    num_workers=8,
)

# [dataloader-end]


# Instantiate Model
class Model(nn.Module):
    """Toy convolutional neural network architecture in pytorch for MNIST."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=0)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 16, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = F.relu(out)
        out = F.adaptive_avg_pool2d(out, (4, 4))
        out = torch.flatten(out, 1, -1)
        out = self.fc1(out)
        out = F.relu(out)
        return self.fc2(out)


model = ComposerClassifier(module=Model(num_classes=10))

# [trainer-start]
# Instantiate the trainer
composer_trace_dir = 'composer_profiler'
torch_trace_dir = 'torch_profiler'

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=train_dataloader,
    max_duration=2,
    device='gpu' if torch.cuda.is_available() else 'cpu',
    eval_interval=0,
    precision='amp' if torch.cuda.is_available() else 'fp32',
    train_subset_num_batches=16,
    profiler=Profiler(
        trace_handlers=[JSONTraceHandler(folder=composer_trace_dir, overwrite=True)],
        schedule=cyclic_schedule(
            wait=0,
            warmup=1,
            active=4,
            repeat=1,
        ),
        torch_prof_folder=torch_trace_dir,
        torch_prof_overwrite=True,
        torch_prof_memory_filename=None,
    ),
)
# [trainer-end]

# [fit-start]
# Run training
trainer.fit()
# [fit-end]
