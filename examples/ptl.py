import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR

from composer.models.base import ComposerModel


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


## for demo simplicity, skipping PTL datamodule as we need extra dataset_train params

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='/localdisk/CIFAR10', train=True, download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='/localdisk/CIFAR10', train=False, download=True, transform=transform)

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True,
                                               num_workers=2)  #cifar10_dm.train_dataloader()
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

from composer import Trainer


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model.cuda()


class LitResnet(LightningModule):

    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // 256
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=30,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


module = create_model()
PTLmodel = LitResnet(lr=0.05)
## up to here all PTL code

## The ComposerModel needs 5 functions: __init__, loss, metrics, forward and validate. Everything else is under the hood.


class Mosaicmodel(ComposerModel):

    def __init__(self) -> None:
        super().__init__()

    def loss(self, outputs, batch, *args, **kwargs):  # -> Tensors:
        ## loss from PTL.training_step()
        x, y = batch
        return F.nll_loss(outputs, y)

    def metrics(self, train):
        ## acc from PTL.evaluate()
        from torchmetrics.classification.accuracy import Accuracy
        return Accuracy()

    def forward(self, batch):
        ## from PTL.forward()
        x, _ = batch
        return PTLmodel.forward(x)

    def validate(self, batch):
        ## from PTL.evaluate()
        return PTLmodel.evaluate(batch, "val")


## COMPOSER TRAINER
trainer = Trainer(model=Mosaicmodel().cuda(),
                  train_dataloader=train_dataloader,
                  eval_dataloader=test_dataloader,
                  optimizers=PTLmodel.configure_optimizers()["optimizer"],
                  max_duration='1ep',
                  device='gpu',
                  validate_every_n_epochs=-1,
                  seed=42)

trainer.fit()
