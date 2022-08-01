# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging  # disable logging in notebook
import os
import sys

logging.disable(sys.maxsize)

import torch
from torch.utils.data import DataLoader

import composer
from composer.core.data_spec import DataSpec
from composer.datasets import coco_mmdet
from composer.datasets.coco_mmdet import mmdet_collate
from composer.models import composer_yolox
from composer.trainer.devices import DeviceGPU

train_dataset = coco_mmdet(path='/workdisk/austin/data/coco', split='train')
val_dataset = coco_mmdet(path='/workdisk/austin/data/coco', split='val')

model = composer_yolox(model_name='yolox-s')

train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=mmdet_collate)
val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=mmdet_collate)

optimizer = composer.optim.DecoupledSGDW(
    model.parameters(),  # Model parameters to update
    lr=0.05,  # Peak learning rate
    momentum=0.9,
    weight_decay=2.0e-3  # If this looks large, it's because its not scaled by the LR as in non-decoupled weight decay
)

lr_scheduler = composer.optim.LinearWithWarmupScheduler(
    t_warmup='1ep',  # Warm up over 1 epoch
    alpha_i=1.0,  # Flat LR schedule achieved by having alpha_i == alpha_f
    alpha_f=1.0)

train_epochs = '3ep'  # Train for 3 epochs because we're assuming Colab environment and hardware

trainer = composer.trainer.Trainer(model=model,
                                   train_dataloader=DataSpec(train_loader, get_num_samples_in_batch=lambda x: 64),
                                   eval_dataloader=DataSpec(val_loader, get_num_samples_in_batch=lambda x: 64),
                                   max_duration=train_epochs,
                                   optimizers=optimizer,
                                   train_subset_num_batches=10,
                                   schedulers=lr_scheduler,
                                   device='gpu' if torch.cuda.is_available() else 'cpu')

trainer.fit()
