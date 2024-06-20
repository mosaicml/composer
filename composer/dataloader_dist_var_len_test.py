from typing import Any

import torch
import torch.nn as nn
from composer import Callback, Event, Logger, State, Trainer
from composer.devices import Device, DeviceCPU, DeviceGPU, DeviceMPS
from composer.models import ComposerModel
from composer.optim import DecoupledAdamW
from composer.utils import dist
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy


# Synthetic binary dataset
class BinaryDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {"features": self.x[idx], "labels": self.y[idx]}


# Single layer NN
class SimpleLinearModel(ComposerModel):

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_metric = BinaryAccuracy(sync_on_compute=True, dist_sync_on_step=False)

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        out: Tensor = self.linear(x["features"])
        return out

    def loss(self, outputs: Tensor, batch: dict[str, Tensor], *args: Any, **kwargs: Any) -> Tensor:
        loss: Tensor = self.loss_fn(outputs, batch["labels"])
        return loss

    def get_metrics(self, is_train: bool = False) -> dict[str, Metric]:
        return {} if is_train else {"accuracy": self.acc_metric}

    def update_metric(self, batch: dict[str, Tensor], outputs: Tensor, metric: Metric) -> None:
        metric.update(outputs.argmax(dim=1), batch["labels"])


# Callback to print all key events to help debugging
class DebugCallback(Callback):

    events = [
        Event.INIT,
        Event.BEFORE_LOAD,
        Event.AFTER_LOAD,
        Event.FIT_START,
        Event.ITERATION_START,
        Event.EPOCH_START,
        Event.EVAL_BEFORE_ALL,
        Event.EVAL_START,
        Event.EVAL_BATCH_START,
        Event.EVAL_BEFORE_FORWARD,
        Event.EVAL_AFTER_FORWARD,
        Event.EVAL_BATCH_END,
        Event.EVAL_END,
        Event.EVAL_AFTER_ALL,
        Event.EPOCH_CHECKPOINT,
        Event.ITERATION_END,
        Event.ITERATION_CHECKPOINT,
        Event.FIT_END,
    ]

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if event in self.events:
            print(f"Event: {event}")


def build_dataloader(num_samples: int, num_features: int) -> DataLoader[dict[str, Tensor]]:
    x = torch.rand((num_samples, num_features))
    y = torch.randint(low=0, high=2, size=(num_samples,))
    dataset = BinaryDataset(x, y)
    dist_sampler = dist.get_sampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=dist_sampler)
    return dataloader


def get_best_accelerator() -> Device:
    if torch.cuda.is_available():
        return DeviceGPU()
    if torch.backends.mps.is_available():
        return DeviceMPS()
    return DeviceCPU()


def run() -> None:
    # Default values for dataset creation
    num_features = 10
    num_train_samples = 512
    num_val_samples = 256
    # Change rank 1 dataloader size
    if dist.get_local_rank() == 1:
        num_train_samples += 256
        num_val_samples += 256
    # Construct everything
    print("Building dataloaders...")
    train_dataloader = build_dataloader(num_train_samples, num_features)
    val_dataloader = build_dataloader(num_val_samples, num_features)
    print(f" Train Dataloader Len: {len(train_dataloader)}")
    print(f"   Val Dataloader Len: {len(val_dataloader)}")
    print("Building model...")
    model = SimpleLinearModel(num_features)
    optimizer = DecoupledAdamW(model.parameters(), lr=1e-3)
    print("Building trainer...")
    trainer = Trainer(
        model=model,
        device=get_best_accelerator(),
        optimizers=optimizer,
        max_duration="2ep",
        log_to_console=True,
        console_log_interval="4ba",
        progress_bar=False,
        dist_timeout=30,
        callbacks=[DebugCallback()],
    )
    # Actually fit (train and eval)
    print("Fitting...")
    trainer.fit(train_dataloader=train_dataloader, eval_dataloader=val_dataloader)


if __name__ == "__main__":
    run()
