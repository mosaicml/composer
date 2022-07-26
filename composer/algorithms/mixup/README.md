# 🥣 MixUp


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

MixUp synthesizes new training examples by combining pairs of individual examples.
For any pair of examples, it trains the network on a random convex combination of the individual examples (i.e., a random interpolation between the two examples).
To create the corresponding targets, it uses the same random convex combination of the targets of the individual examples.
Training in this fashion improves generalization.

| ![MixUp](https://storage.googleapis.com/docs.mosaicml.com/images/methods/mix_up.png) |
|:--:
|*Two different training examples (a picture of a bird and a picture of a frog) that have been combined by MixUp into a single example. The corresponding targets are a convex combination of the targets for the bird class and the frog class.*|

## How to Use

### Functional Interface

Here we run `mixup` using index labels and interpolating the loss (a trick when using cross entropy).
```python
import torch
import torch.nn.functional as F
import composer.functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            X_mixed, y_perm, mixing = cf.mixup_batch(X, y, alpha=0.2)
            y_hat = model(X_mixed)
            loss = (1 - mixing) * loss_fn(y_hat, y) + mixing * loss_fn(y_hat, y_perm)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

Here we run `mixup` using dense/one-hot labels and interpolate the labels (general case).
```python
import torch
import torch.nn.functional as F
import composer.functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            X_mixed, y_perm, mixing = cf.mixup_batch(X, y, alpha=0.2)
            y_mixed = (1 - mixing) * y + mixing * y_perm
            y_hat = model(X_mixed)
            loss = loss_fn(y_hat, y_mixed)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

Here we run `mixup` using index labels and interpolate the loss (a trick when using cross entropy)

<!--pytest.mark.gpu-->
<!--
```python
from torch.utils.data import DataLoader
from tests.common import RandomClassificationDataset, SimpleModel

model = SimpleModel()
train_dataloader = DataLoader(RandomClassificationDataset())
```
-->
<!--pytest-codeblocks:cont-->
```python
from composer.algorithms import MixUp
from composer.trainer import Trainer

mixup = MixUp(
    alpha=0.2,
    interpolate_loss=True
)

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    max_duration='1ep',
    algorithms=[mixup]
)

trainer.fit()
```

Here we run `mixup` using dense/one-hot labels and interpolate the labels (general case).

<!--pytest.mark.gpu-->
<!--
```python
from torch.utils.data import DataLoader
from tests.common import RandomClassificationDataset, SimpleModel

model = SimpleModel()
train_dataloader = DataLoader(RandomClassificationDataset())
eval_dataloader = DataLoader(RandomClassificationDataset())
```
-->
<!--pytest-codeblocks:cont-->
```python
from composer.algorithms import MixUp
from composer.trainer import Trainer

mixup = MixUp(
    alpha=0.2,
    interpolate_loss=False
)

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep',
    algorithms=[mixup]
)

trainer.fit()
```

### Implementation Details

Composer's MixUp has two modes of use. When using `interpolate_loss=True`, MixUp does not directly interpolate the targets but rather interpolates the loss function by calling it on the original target and the mixed-in target and interpolating the two loss values. For loss functions that are linear in the targets (such as cross entropy), this is mathematically equivalent to interpolating the targets. Many other implementations of MixUp work this way.

For loss functions that are not linear in the targets, this trick may still be used but results in behavior that differs from the description in the [original paper](https://arxiv.org/abs/1710.09412). If the loss function in use can accept non-index targets (such as dense or one-hot labels), then Composer's MixUp can be used with `interpolate_loss=False`, which interpolates the targets of the two samples. This gives behavior matching the usual description of MixUp.

## Suggested Hyperparameters

The sole hyperparameter for MixUp is `alpha`, which controls the Beta distribution that determines the extent of interpolation between the two input examples.
Concretely, for two examples `x1` and `x2`, the example that MixUp produces is `t * x1 + (1 - t) * x2` where `a` is sampled from the distribution `Beta(alpha, alpha)`.
On ImageNet, we found that `alpha=0.2` (which places slightly higher probability on extreme values of `t`) works well.
On CIFAR-10, we found that `alpha=1.0` (which samples `t` uniformly between 0 and 1) works well.


## Technical Details

Mixed samples are created from a batch `(X1, y1)` of (inputs, targets) together with version `(X2, y2)` where the ordering of examples has been shuffled. The examples can be mixed by sampling a value `t` (between 0.0 and 1.0) from the Beta distribution parameterized by `alpha` and training the network on the interpolation between `(X1, y1)` and `(X2, y2)` specified by `t`. That is, the batch of examples produced by MixUp is `t * X1 + (1 - t) * X2`, and the batch of targets produced by MixUp is `t * y1 + (1 - t) * y2`.

We set `(X2, y2)` to be a shuffling of the original batch `(X1, y1)` (rather than sampling an entirely new batch) to allow MixUp to proceed without loading additional data.
This choice avoids putting additional strain on the dataloader.


Data augmentation techniques can sometimes put additional load on the CPU, potentially reaching the point where the CPU becomes a bottleneck for training.
To prevent this from happening for MixUp, our implementation of MixUp (1) occurs on the GPU and (2) uses the same `t` for all examples in the minibatch.
Doing so avoids putting additional work on the CPU (since augmentation occurs on the GPU) and minimizes additional work on the GPU (since all images are handled uniformly within a batch).

> 🚧 MixUp Requires a Small Amount of Additional GPU Compute and Memory
>
> MixUp requires a small amount of additional GPU compute and memory to produce the mixed-up batch.
> In our experiments, we have found these additional resource requirements to be negligible.

> ❗ When `interpolate_loss=False`, MixUp Produces a Full Distribution, Not a Target Index
>
> Many classification tasks represent the target value using the index of the target value rather than the full one-hot encoding of the label value.
> With `interpolate_loss=False`, MixUp interpolates between two target values for each example. It must represent the final targets as a dense vector of probabilities.
> Our implementation of MixUp turns each label into a dense vector of probabilities (if it has not already been converted into a distribution).
> The loss function used for the model must be able to accept this dense vector of probabilities as the target.

>❗ When `interpolate_loss=True` MixUp interpolates the loss rather than the targets.
>
> This is fine for loss functions that are linear in the targets, such as cross entropy, but may produce unexpected results for other loss functions.

> 🚧 Composing Regularization Methods
>
> As a general rule, composing regularization methods may lead to diminishing returns in quality improvements. MixUp is one such regularization method.

Although our implementation of MixUp is designed for computer vision tasks, variants of MixUp have been shown to be useful in other settings, for example, natural language processing.
The implementation in Composer currently only supports computer vision.


## Attribution

[*mixup: Beyond Empirical Risk Minimization*](https://arxiv.org/abs/1710.09412) by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. Published in ICLR 2018.

*This Composer implementation of this method and the accompanying documentation were produced by Cory Stephenson at MosaicML.*
