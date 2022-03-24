# Copyright 2021 MosaicML. All Rights Reserved.

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer import Trainer
from composer.models import MNIST_Classifier
from composer.profiler import JSONTraceHandler, cyclic_schedule

# Specify Dataset and Instantiate DataLoader
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
model = MNIST_Classifier(num_classes=10)

# Instantiate the trainer
composer_trace_dir = "composer_profiler"
torch_trace_dir = "torch_profiler"

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=train_dataloader,
                  max_duration=2,
                  device="gpu",
                  validate_every_n_batches=-1,
                  validate_every_n_epochs=-1,
                  precision="amp",
                  train_subset_num_batches=16,
                  prof_trace_handlers=JSONTraceHandler(folder_format=composer_trace_dir, overwrite=True),
                  prof_schedule=cyclic_schedule(
                      wait=0,
                      warmup=1,
                      active=4,
                      repeat=1,
                  ),
                  torch_prof_folder_format=torch_trace_dir,
                  torch_prof_overwrite=True)

# Run training
trainer.fit()
