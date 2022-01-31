# Copyright 2021 MosaicML. All Rights Reserved.

"""How to BYOT (bring your own trainer).

Allows algorithms to be used with minimal dependencies on our repo.

Example invocation::

    >>> python byo_trainer.py --datadir /datasets/
"""
import argparse
import logging

import torch
import torch.utils.data
import torchmetrics
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification.accuracy import Accuracy
from torchvision import datasets, transforms

import composer
from composer import Event
from composer.algorithms import BlurPool
from composer.core.types import Evaluator, Precision
from composer.utils import ensure_tuple

logging.basicConfig()
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True, help='data dir for MNIST')
parser.add_argument('--train_batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=5, help='epochs')

args = parser.parse_args()


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=0)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 16, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = F.relu(out)
        out = F.adaptive_avg_pool2d(out, (4, 4))
        out = torch.flatten(out, 1, -1)
        out = self.fc1(out)
        out = F.relu(out)
        return self.fc2(out)


def train():

    model = Model().cuda()
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(args.datadir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(args.datadir, train=False, download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
    )

    # Wrap your validation set in an Evaluator with relevant metrics
    # Leaving metrics as None will use the default validation metrics in your model
    evaluator = Evaluator(label='eval_dataset', dataloader=val_dataloader, metrics=torchmetrics.Accuracy())

    # to use our algorithms, create and maintain the trainer state
    state = composer.State(
        model=model,
        train_dataloader=train_dataloader,
        evaluators=[evaluator],
        max_duration=f"{args.epochs}ep",
        grad_accum=1,
        precision=Precision.FP32,
    )

    # define which algorithms to apply and configure the cmp engine
    state.algorithms = [BlurPool(replace_convs=True, replace_maxpools=True, blur_first=False)]

    engine = composer.Engine(state=state)

    # add two-way callbacks in the trainer loop
    engine.run_event(
        Event.INIT)  # Event.INIT should be run BEFORE any DDP fork and before optimizers and schedulers are created

    state.optimizers = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)
    state.schedulers = torch.optim.lr_scheduler.MultiStepLR(state.optimizers, milestones=[2, 4], gamma=0.1)

    engine.run_event(Event.TRAINING_START)  # the Event.TRAINING_START should be run AFTER any DDP fork

    while state.timer < state.max_duration:
        logging.info(f'Epoch {state.epoch}')
        engine.run_event(Event.EPOCH_START)

        for x, y in train_dataloader:
            engine.run_event(Event.BATCH_START)

            state.batch = x.cuda(), y.cuda()
            engine.run_event(Event.AFTER_DATALOADER)
            engine.run_event(Event.BEFORE_TRAIN_BATCH)
            engine.run_event(Event.BEFORE_FORWARD)
            state.outputs = model(state.batch)
            engine.run_event(Event.AFTER_FORWARD)
            engine.run_event(Event.BEFORE_LOSS)
            state.loss = F.cross_entropy(input=state.outputs, target=state.last_target)  # type: ignore
            engine.run_event(Event.AFTER_LOSS)
            engine.run_event(Event.BEFORE_BACKWARD)
            state.loss.backward()
            engine.run_event(Event.AFTER_BACKWARD)

            for optimizer in ensure_tuple(state.optimizers):
                optimizer.step()
            engine.run_event(Event.AFTER_TRAIN_BATCH)
            for optimizer in ensure_tuple(state.optimizers):
                optimizer.zero_grad()

            if state.timer.batch.value % 100 == 0:
                logging.info(f'Epoch {state.epoch}, Step: {state.timer.batch.value}, loss: {state.loss:.3f}')
            engine.run_event(Event.BATCH_END)
            state.timer.on_batch_complete(len(state.batch))

        metric = Accuracy().cuda()
        for (x, y) in val_dataloader:
            x = x.cuda()
            y = y.cuda()

            x = model(x)
            prediction = torch.argmax(x, dim=1)

            metric(prediction, y)
        logging.info(f'Epoch {state.epoch} complete. val/acc = {metric.compute():.5f}')

        engine.run_event(Event.EPOCH_END)
        state.timer.on_epoch_complete()

    engine.run_event(Event.TRAINING_END)


if __name__ == '__main__':
    train()
