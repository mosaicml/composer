# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Save and Load Checkpoints with `Weights and Biases <https://wandb.ai/>`."""

import shutil

import torch.utils.data
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from composer import Trainer
from composer.loggers import WandBLogger
from composer.models.classify_mnist import mnist_model

# Configure the WandBLogger to log artifacts, and set the project name
# The project name must be deterministic, so we can restore from it
wandb_logger = WandBLogger(
    log_artifacts=True,
    project='my-wandb-project-name',
)

# Configure the trainer -- here, we train a simple MNIST classifier
print('Starting the first training run\n')
model = mnist_model(num_classes=10)
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
    save_folder='/tmp/checkpoints',  # The trainer requires that checkpoints must be saved locally before being upload
    save_interval='1ep',
    save_artifact_name='epoch{epoch}.pt',  # Name checkpoints like epoch1.pt, epoch2.pt, etc...
    save_num_checkpoints_to_keep=0,  # Do not keep any checkpoints locally after they have been uploaded to W & B
)

# Train!
trainer.fit()

# Remove the temporary folder to ensure that the checkpoint is downloaded from the cloud
shutil.rmtree('/tmp/checkpoints', ignore_errors=True)

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
    load_path='epoch1.pt:latest',  # Load the checkpoint -- WandB requires that the load_path include the "latest" tag

    #  (Optional) Checkpoint Saving Configuration to continue to save new checkpoints
    loggers=wandb_logger,  # Log checkpoints via the WandB Logger
    save_folder='/tmp/checkpoints',  # The trainer requires that checkpoints must be saved locally before being upload
    save_interval='1ep',
    save_artifact_name='epoch{epoch}.pt',  # Name checkpoints like epoch1.pt, epoch2.pt, etc...
    save_num_checkpoints_to_keep=0,  # Do not keep any checkpoints locally after they have been uploaded to W & B
)

# Verify that we loaded the checkpoint. This should print 1ep, since we already trained for 1 epoch.
print(f'\nResuming training at epoch {trainer.state.timestamp.epoch}\n')

# Train for another epoch!
trainer.fit()
