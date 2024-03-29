# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Save and Load Checkpoints with `Weights and Biases <https://wandb.ai/>`."""

import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from composer import Trainer
from composer.loggers import WandBLogger
from composer.models.tasks import ComposerClassifier

# Configure the WandBLogger to log artifacts, and set the project name
# The project name must be deterministic, so we can restore from it
wandb_logger = WandBLogger(
    log_artifacts=True,
    project='my-wandb-project-name',
)

# Configure the trainer -- here, we train a simple MNIST classifier
print('Starting the first training run\n')


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
optimizer = SGD(model.parameters(), lr=0.01)
train_dataloader = torch.utils.data.DataLoader(
    dataset=MNIST('~/datasets', train=True, download=True, transform=ToTensor()),
    batch_size=2048,
)
eval_dataloader = torch.utils.data.DataLoader(
    dataset=MNIST('~/datasets', train=True, download=True, transform=ToTensor()),
    batch_size=2048,
)
trainer = Trainer(
    model=model,
    max_duration='1ep',
    optimizers=optimizer,

    # Train Data Configuration
    train_dataloader=train_dataloader,
    train_subset_num_batches=5,  # For this example, limit each epoch to 5 batches

    # Evaluation Configuration
    eval_dataloader=eval_dataloader,
    eval_subset_num_batches=5,  # For this example, limit evaluation to 5 batches

    # Checkpoint Saving Configuration
    loggers=wandb_logger,  # Log checkpoints via the WandB Logger
    save_folder='checkpoints',  # This the folder that checkpoints are saved to locally and remotely.
    save_interval='1ep',
    save_filename='epoch{epoch}.pt',  # Name checkpoints like epoch1.pt, epoch2.pt, etc...
    save_num_checkpoints_to_keep=0,  # Do not keep any checkpoints locally after they have been uploaded to W & B
)

# Train!
trainer.fit()

# Remove the temporary folder to ensure that the checkpoint is downloaded from the cloud
shutil.rmtree('checkpoints', ignore_errors=True)

# Close the existing trainer to trigger W & B to mark the run as "finished", and be ready for the next training run
trainer.close()

# Construct a new trainer that loads from the previous checkpoint
print('\nStarting the second training run\n')
trainer = Trainer(
    model=model,
    max_duration='2ep',  # Train to 2 epochs in total
    optimizers=optimizer,

    # Train Data Configuration
    train_dataloader=train_dataloader,
    train_subset_num_batches=5,  # For this example, limit each epoch to 5 batches

    # Evaluation Configuration
    eval_dataloader=eval_dataloader,
    eval_subset_num_batches=5,  # For this example, limit evaluation to 5 batches

    # Checkpoint Loading Configuration
    load_object_store=wandb_logger,
    # Load the checkpoint using the save_folder plus the save_filename -- WandB requires that the load_path include the "latest" tag
    load_path='checkpoints/epoch1.pt:latest',
    #  (Optional) Checkpoint Saving Configuration to continue to save new checkpoints
    loggers=wandb_logger,  # Log checkpoints via the WandB Logger
    save_folder='checkpoints',  # The trainer requires that checkpoints must be saved locally before being uploaded
    save_interval='1ep',
    save_filename='epoch{epoch}.pt',  # Name checkpoints like epoch1.pt, epoch2.pt, etc...
    save_num_checkpoints_to_keep=0,  # Do not keep any checkpoints locally after they have been uploaded to W & B
)

# Verify that we loaded the checkpoint. This should print 1ep, since we already trained for 1 epoch.
print(f'\nResuming training at epoch {trainer.state.timestamp.epoch}\n')

# Train for another epoch!
trainer.fit()
