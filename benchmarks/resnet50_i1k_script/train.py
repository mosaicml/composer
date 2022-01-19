# Copyright 2021 MosaicML. All Rights Reserved.

from composer.optim.optimizer_hparams import DecoupledSGDWHparams
import torch
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import Compose, Resize, RandomResizedCrop, RandomHorizontalFlip, CenterCrop, Normalize, ToTensor

import composer.models
from composer import Trainer
from composer.optim import DecoupledSGDWHparams, CosineAnnealingLRHparams, WarmUpLRHparams
# Example algorithms to train with
from model import ImageNet_ResNet

# Data config - Imagenet
resize_size= 256
crop_size = 224
datadir= '/datasets/ImageNet'
drop_last= True


# optimizer config - Decoupled SGDW
lr =  2.048
momentum =  0.875
weight_decay = 5.0e-4
dampening = 0
nesterov= False


# Scheduler configs - warmup
warmup_iters =  "8ep"
warmup_method =  'linear'
warmup_factor =  10

# cosine
T_max =  "82ep"
eta_min =  0

# model config
    # initializers:
    #   - kaiming_normal
    #   - bn_uniform
    #
num_classes= 1000
# loggers:
#   - tqdm: {}

#trainer config
max_duration =  '90ep'
train_batch_size= 2048
eval_batch_size= 2048
seed= 17

#   gpu: {}
# dataloader:
#   pin_memory: true
#   timeout: 0
#   prefetch_factor: 2
#   persistent_workers: true
#   num_workers: 8
validate_every_n_epochs=1
grad_accum= 1
precision='amp'




IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)

# Your custom model
class ResNet(composer.models.MosaicClassifier):

    def __init__(self, num_classes: int):
        super().__init__(module=ImageNet_ResNet(num_classes=num_classes))


# augmentations 

train_transforms = Compose([Normalize(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD),
                            Resize(resize_size),
                            RandomResizedCrop(crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
                            RandomHorizontalFlip(),
                            ToTensor()])

val_transforms = Compose([Normalize(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD),
                        Resize(resize_size),
                        CenterCrop(crop_size),
                        ToTensor()])


# Your custom train dataloader
train_dataloader = torch.utils.data.DataLoader(dataset=datasets.ImageFolder(datadir, transform=train_transforms),
    drop_last=True,
    shuffle=True,
    batch_size=train_batch_size)

# Your custom eval dataloader
eval_dataloader = torch.utils.data.DataLoader(dataset=datasets.ImageFolder(datadir, transform=val_transforms),
    drop_last=False,
    shuffle=False,
    batch_size=eval_batch_size)


optimizer = DecoupledSGDWHparams(lr=lr, momentum=momentum, weight_decay=weight_decay)
lr_schedule = [WarmUpLRHparams(warmup_iters=warmup_iters, warmup_method=warmup_method, warmup_factor=warmup_factor, interval='step'),
               CosineAnnealingLRHparams(T_max=T_max, eta_min=eta_min, interval='step')]

model = ResNet(num_classes=num_classes)
# Initialize Trainer with custom model, custom train and eval datasets, and algorithms to train with
trainer = Trainer(model=ResNet(num_classes=num_classes),
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration=max_duration,
                  optimizer_hparams=optimizer,
                  schedulers_hparams=lr_schedule
                  seed=seed,
                  grad_accum=grad_accum,
                  precision=precision,
                  validate_every_n_epochs=validate_every_n_epochs
                  )

trainer.fit()