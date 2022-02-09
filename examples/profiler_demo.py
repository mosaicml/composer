# Copyright 2021 MosaicML. All Rights Reserved.

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import composer

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
                                   train_subset_num_batches=train_subset_num_batches,
                                   profiling=True,
                                   prof_skip_first=0,
                                   prof_wait=0,
                                   prof_warmup=1,
                                   prof_active=4,
                                   prof_repeat=1)

# Run training
trainer.fit()
