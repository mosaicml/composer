# 🥙 Gyro Dropout
[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Computer Vision`

Gyro Dropout replaces implementations of `torch.nn.Dropout`. The Gyro Dropout provides increased accuracy compared with dropout.

Gyro dropout is a variant of dropout that improves the efficiency of training neural networks.
Instead of randomly dropping out neurons in every training iteration, gyro dropout pre-selects and trains a fixed
number of subnetworks. 'Sigma' is the number of total pre-selected subnetworks and 'Tau' is the number of concurrently scheduled subnetworks in an iteration.

## How to Use

### Functional Interface

```python
# Apply surgery on the model to swap-in the Gyro Dropout using the Composer functional API

import composer.functional as cf

def training_loop(model, train_loader):
    cf.apply_gyro_dropout(
        model,
        iters_per_epoch = 196,
        max_epoch = 100,
        p = 0.5,
        sigma = 256,
        tau = 16,
    )

    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for X, y in train_loader:
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
```

### Composer Trainer

<!--pytest.mark.gpu-->
<!--pytest.mark.filterwarnings('ignore:GyroDropout is not implemented in a way that allows correct resumption.*:UserWarning')-->
<!--
```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Metric, MetricCollection

from composer.models import ComposerClassifier

from typing import Any, Callable, Dict, Optional, Tuple, Union


class SimpleDataset(Dataset):

    def __init__(self, size: int = 256, batch_size: int = 256, feature_size: int = 1, num_classes: int = 2):
        self.size = size
        self.batch_size = batch_size
        self.x = torch.randn(size*batch_size, feature_size)
        self.y = torch.randint(0, num_classes, size=(size*batch_size,), dtype=torch.long)

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.x[index*self.batch_size:(index+1)*self.batch_size], self.y[index*self.batch_size:(index+1)*self.batch_size]


class SimpleModelWithDropout(ComposerClassifier):

    def __init__(self, num_features: int = 64, num_classes: int = 10) -> None:
        fc1 = torch.nn.Linear(num_features, 512)
        fc2 = torch.nn.Linear(512, num_classes)
        dropout = torch.nn.Dropout(0.5)

        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            fc1,
            torch.nn.ReLU(),
            dropout,
            fc2,
            torch.nn.Softmax(dim=-1),
        )

        super().__init__(module=net, num_classes=num_classes)

        self.fc1 = fc1
        self.fc2 = fc2

    def loss(self, outputs: torch.Tensor, batch: Tuple[Any, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        _, targets = batch
        targets = targets.squeeze(dim=0)
        # import sys
        # print(f"output shape: {outputs.shape}\n", file=sys.stderr)
        # print(f"target shape: {targets.shape}\n", file=sys.stderr)
        return self._loss_fn(outputs, targets, *args, **kwargs)

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        _, targets = batch
        metric.update(outputs.squeeze(dim=0), targets.squeeze(dim=0))

    def forward(self, batch: Tuple[torch.Tensor, Any]) -> torch.Tensor:
        inputs, _ = batch
        inputs = inputs.squeeze(dim=0)
        outputs = self.module(inputs)
        return outputs


model = SimpleModelWithDropout()
train_dataloader = DataLoader(SimpleDataset(batch_size=256, feature_size=64, num_classes=10))
eval_dataloader = DataLoader(SimpleDataset(batch_size=256, feature_size=64, num_classes=10))
```
-->
<!--pytest-codeblocks:cont-->
```python

from composer.algorithms import GyroDropout
from composer.trainer import Trainer

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='1ep',
                  algorithms=[GyroDropout(p=0.5,
                                          sigma=256,
                                          tau=16)])

trainer.fit()
```

### Implementation Details

Gyro Dropout is implemented by performing model surgery, which looks for instances of `torch.nn.Dropout`. This should be applicable to any model that utilizes `torch.nn.Dropout`.

## Suggested Hyperparameters

Gyro Dropout has two hyperparameters - sigma, tau. (iters_per_epoch and max_epoch is training-dependent)

For the hyperparameters (sigma, tau), we recommend (256, 16) in AlexNet, LeNet or (1024, 8) in ResNet-18, BERT.


## Technical Details
GyroDropout achieves improved accuracy over conventional dropout by pre-selecting a fixed number of subnetworks and training with only those subnetworks. Because the selected subnetworks are trained more robustly (compared to the conventional dropout), their diversity increases and thus their ensemble achieves higher accuracy.

## Attribution
[*Gyro Dropout: Maximizing Ensemble Effect in Neural Network Training*](https://proceedings.mlsys.org/paper/2022/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract.html) by Junyeol Lee, Hyeongju Kim, Hyungjun Oh, Jaemin Kim, Hongseok Jeung, Yung-Kyun Noh, Jiwon Seo.

*The Composer implementation of this method and the accompanying documentation were produced by Junyeol Lee and Gihyun Park at BDSL in Hanyang Univ.*
