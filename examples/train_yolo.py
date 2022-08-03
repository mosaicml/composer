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
from composer.datasets.coco_mmdet import mmdet_collate, mmdet_get_num_samples
from composer.models import composer_yolox
from composer.loggers import InMemoryLogger, LogLevel, WandBLogger


coco_path = '../data/coco'
batch_size = 16
num_workers = 8


train_dataset = coco_mmdet(path=coco_path, split='train')
val_dataset = coco_mmdet(path=coco_path, split='val')

model = composer_yolox(model_name='yolox-s')

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=mmdet_collate, shuffle=True, drop_last=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=mmdet_collate, num_workers=num_workers)

optimizer = composer.optim.DecoupledSGDW(
    model.parameters(), # Model parameters to update
    lr=0.01, # Peak learning rate
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True # If this looks large, it's because its not scaled by the LR as in non-decoupled weight decay
)

lr_scheduler = composer.optim.CosineAnnealingWithWarmupScheduler(
    t_warmup="1ep", # Warm up over 30 epoch
)

train_epochs = '3ep'  # Train for 3 epochs because we're assuming Colab environment and hardware

trainer = composer.trainer.Trainer(model=model,
                                   train_dataloader=DataSpec(train_loader, get_num_samples_in_batch=mmdet_get_num_samples),
                                   eval_dataloader=DataSpec(val_loader, get_num_samples_in_batch=mmdet_get_num_samples),
                                   max_duration=train_epochs,
                                   optimizers=optimizer,
                                   train_subset_num_batches=10,
                                   schedulers=lr_scheduler,
                                   device='gpu' if torch.cuda.is_available() else 'cpu',
                                    loggers=[InMemoryLogger(log_level=LogLevel.BATCH), WandBLogger(project='yolox-test')])


trainer.fit()
