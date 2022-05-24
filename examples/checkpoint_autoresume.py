# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Training with Checkpoint Autoresumption.

In this example, the Trainer is configured with `autoresume` set to True. This flag
instructs the trainer to look in the `save_folder` for an existing checkpoint before starting
a new training run.

To see this example in action, run this script twice.

* The first time the script is run, the trainer will save a checkpoint to the `save_folder` and train
  for one epoch.
* Any subsequent time the script is run, the trainer will resume from where the latest checkpoint. If
  the latest checkpoint was saved at ``max_duration``, meaning all training is finished, the Trainer will
  exit immediately with an error that no training would occur.

To simulate a flaky spot instance, try terminating the script (e.g. Ctrl-C) midway through the
first training run (say, after epoch 0 is finished). Notice how the progress bars would resume at the next
epoch and not repeat any training already completed.

This feature does not require code or configuration changes to distinguish between starting a new training
run or automatically resuming from an existing one, making it easy to use Composer on preemptable cloud instances.
Simply configure the instance to start Composer with the same command every time until training has finished!
"""

import torch.utils.data
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from composer import Trainer
from composer.models.classify_mnist import MNIST_Classifier

# Configure the trainer -- here, we train a simple MNIST classifier
model = MNIST_Classifier(num_classes=10)
optimizer = SGD(model.parameters(), lr=0.01)
train_dataloader = torch.utils.data.DataLoader(
    dataset=MNIST("~/datasets", train=True, download=True, transform=ToTensor()),
    batch_size=2048,
)
eval_dataloader = torch.utils.data.DataLoader(
    dataset=MNIST("~/datasets", train=True, download=True, transform=ToTensor()),
    batch_size=2048,
)

# When using `autoresume`, it is required to specify the `run_name` is required, so
# Composer will know which training run to resume
run_name = "my_autoresume_training_run"

trainer = Trainer(
    model=model,
    max_duration="5ep",
    optimizers=optimizer,

    # Train Data Configuration
    train_dataloader=train_dataloader,
    train_subset_num_batches=5,  # For this example, limit each epoch to 5 batches

    # Evaluation Configuration
    eval_dataloader=eval_dataloader,
    eval_subset_num_batches=5,  # For this example, limit evaluation to 5 batches

    # Checkpoint Configuration
    run_name=run_name,
    save_folder="./my_autoresume_training_run",
    save_interval="1ep",

    # Configure autoresume!
    autoresume=True,
)

print("Training!")

# Train!
trainer.fit()

# Print the number of trained epochs (should always bee the `max_duration`, which is 5ep)
print(f"\nNumber of epochs trained: {trainer.state.timestamp.epoch}")
