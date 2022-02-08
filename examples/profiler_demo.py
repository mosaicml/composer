# Copyright 2021 MosaicML. All Rights Reserved.

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import composer
from composer.profiler import Profiler

# Specify Dataset and Instantiate Dataloader
batch_size = 2048
data_directory = "../data"

mnist_transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(data_directory, train=True, download=True, transform=mnist_transforms)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=True,
                              pin_memory=True,
                              persistent_workers=True,
                              num_workers=8)

# Instantiate Model
model = composer.models.MNIST_Classifier(num_classes=10)

# Instantiate profiler and profiling window
profiler = Profiler(skip_first=0, wait=0, warmup=1, active=4, repeat=1)

# Instantiate the trainer
train_epochs = "2ep"
train_subset_num_batches = 8
device = composer.trainer.devices.DeviceGPU()

trainer = composer.trainer.Trainer(model=model,
                                   train_dataloader=train_dataloader,
                                   eval_dataloader=train_dataloader,
                                   max_duration=train_epochs,
                                   device=device,
                                   validate_every_n_batches=-1,
                                   validate_every_n_epochs=-1,
                                   precision="amp",
                                   profiler=profiler,
                                   train_subset_num_batches=train_subset_num_batches)

# Run training
trainer.fit()
