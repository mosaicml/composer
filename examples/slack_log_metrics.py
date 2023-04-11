# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Example using slack logger to output metrics to slack channel."""

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer import Trainer
from composer.algorithms import ChannelsLast, CutMix, LabelSmoothing
from composer.loggers import SlackLogger
from composer.models import mnist_model

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=128)

SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL', None)

trainer = Trainer(
    model=mnist_model(num_classes=10),
    train_dataloader=train_dataloader,
    max_duration='2ep',
    algorithms=[
        LabelSmoothing(smoothing=0.1),
        CutMix(alpha=1.0),
        ChannelsLast(),
    ],
    loggers=[
        SlackLogger(webhook_url=SLACK_WEBHOOK_URL,
                    log_metrics_formatter_func=(lambda data, **kwargs: [{
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': f'*{k}:* {v}'
                        }
                    } for k, v in data.items()]))
    ],
)

trainer.fit()
