from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from composer import Trainer
from composer.models import mnist_model
from composer.algorithms import LabelSmoothing, CutMix, ChannelsLast

from composer.loggers import WandBLogger, ConsoleLogger, SlackLogger

import os

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=128)

SLACK_WEBHOOK_URL=os.getenv('SLACK_WEBHOOK_URL')

trainer = Trainer(
    model=mnist_model(num_classes=10),
    train_dataloader=train_dataloader,
    max_duration="2ep",
    algorithms=[
        LabelSmoothing(smoothing=0.1),
        CutMix(alpha=1.0),
        ChannelsLast(),
        ],
    loggers=[
        SlackLogger(
            webhook_url=SLACK_WEBHOOK_URL,
            log_metrics_fun=(lambda data, **kwargs:
                [
                    {
                        "type": "section", "text": {"type": "mrkdwn", "text": f"*{k}:* {v}"}
                    }
                    for k, v in data.items()
                ])
        )
    ],
)

trainer.fit()