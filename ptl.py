import torch
from torchvision import datasets, transforms

torch.manual_seed(42)

import matplotlib.pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule

def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=train_transforms)
test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=test_transforms)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=True)

from composer import Trainer


from composer.models import MosaicResNet

module = create_model()
model = LitResnet(lr=0.05)

## change model to composer version

model = models.CIFAR10_ResNet56(num_classes=10)


trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=test_dataloader,
                  max_duration='1ep',
                  device='gpu',
                  validate_every_n_epochs=-1,
                  seed=42)

trainer.fit()


