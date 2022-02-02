import torch
from torchvision import datasets, transforms

torch.manual_seed(42)

import matplotlib.pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

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

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

## up to here all PTL code

train_dataloader = cifar10_dm.train_dataloader()
test_dataloader = cifar10_dm.test_dataloader()

from composer import Trainer


from composer.models import MosaicResNet

module = create_model()
PTLmodel = LitResnet(lr=0.05)


## The MosaicClassifier needs 5 functions: __init__, loss, metrics, forward and validate. Everything else is under the hood.
## change model to composer version
class Mosaicmodel(BaseMosaicModel):

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()

    def loss(self, outputs: Any, batch: BatchPair, *args, **kwargs) -> Tensors:
        ## loss from PTL.training_step()
        return F.nll_loss(logits, y)

    def metrics(self, train: bool = False) -> Metrics:
        ## acc from PTL.evaluate()
        from torchmetrics.classification.accuracy import Accuracy
        return Accuracy()

    def forward(self, batch: BatchPair) -> Tensor:
        ## from PTL.forward()
        x, _ = batch
        return PTLmodel.forward(x)

    def validate(self, batch: BatchPair) -> Tuple[Any, Any]:
        ## from PTL.evaluate()
        return PTLmodel.evaluate(batch, "val")



#model = models.CIFAR10_ResNet56(num_classes=10)


## COMPOSER TRAINER
trainer = Trainer(model=Mosaicmodel,
                  train_dataloader=train_dataloader,
                  eval_dataloader=test_dataloader,
                  max_duration='1ep',
                  device='gpu',
                  validate_every_n_epochs=-1,
                  seed=42)

trainer.fit()


