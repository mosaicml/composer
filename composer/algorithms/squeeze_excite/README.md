# 🫀 Squeeze-and-Excitation

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

Adds a channel-wise attention operator in CNNs. Attention coefficients are produced by a small, trainable MLP that uses the channels' globally pooled activations as input. It requires more work on each forward pass, slowing down training and inference, but leads to higher quality models.

| ![Squeeze-Excite](https://storage.googleapis.com/docs.mosaicml.com/images/methods/squeeze-and-excitation.png) |
|:--|
| *After an activation tensor **X** is passed through Conv2d **F**<sub>tr</sub> to yield a new tensor **U**, a Squeeze-and-Excitation (SE) module scales the channels in a data-dependent manner. The scales are produced by a single-hidden-layer, fully-connected network whose input is the global-averaged-pooled **U**. This can be seen as a channel-wise attention mechanism.* |

## How to Use

### Functional Interface

```python
# Run the Squeze-and-Excitation algorithm directly on the model using the Composer functional API

import torch
import torch.nn.functional as F
import composer.functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())

    # only need to pass in opt if apply_squeeze_excite is used after
    # optimizer creation; otherwise only the model needs to be passed in
    cf.apply_squeeze_excite(
        model,
        optimizers=opt,
        min_channels=128,
        latent_channels=64
    )

    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(1):
        for X, y in train_loader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

<!--pytest.mark.gpu-->
<!--
```python
from torch.utils.data import DataLoader
from tests.common import RandomImageDataset, SimpleConvModel

model = SimpleConvModel()
train_dataloader = DataLoader(RandomImageDataset())
eval_dataloader = DataLoader(RandomImageDataset())
```
-->
<!--pytest-codeblocks:cont-->
```python
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate point in the training loop

from composer.algorithms import SqueezeExcite
from composer.trainer import Trainer

algo = SqueezeExcite(
    min_channels=128,
    latent_channels=64
)

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='10ep',
    algorithms=[algo]
)

trainer.fit()
```

## Implementation Details

In order to be architecture-agnostic, our implementation applies the SE attention mechanism after individual Conv2d modules, rather than at particular points in particular networks. This results in more SE modules being present than in the original paper.

Our implementation also allows applying the SE module after only certain Conv2d modules based on their channel count (see the hyperparameter discussion).


## Suggested Hyperparameters

Squeeze-Excite has two hyperparameters:

- `latent_channels` - The number of channels to use in the hidden layer of the MLP that computes channel attention coefficients
- `min_channels` - The minimum number of output channels in a Conv2d required for an SE module to be added afterward

We recommend setting `latent_channels` to a value such that the minimimum channel count in any layer will be at least 64. One can accomplish this either by 1) directly setting `latent_channels=64` or more, or 2) by setting `latent_channels=r` and `min_channels=int(64/r)` or more, for some `r > 0`.

We recommend setting `min_channels` to `512` for ImageNet-scale or larger models. For smaller models, we recommend using the largest output channel count of any Conv2d in the network. Restricting the application to modules with higher channel counts is beneficial because, in most architectures, higher channel counts are used with smaller spatial sizes. Applying squeeze-excite has the least overhead when the Conv2d's output has a small spatial size (e.g., 14x14 instead of 64x64).

## Technical Details

This method tends to consistently improve the accuracy of CNNs both in absolute terms and when controlling for training and inference time. This may come at the cost of a roughly 20% increase in inference latency, depending on the architecture and inference hardware.

>  ✅ Squeeze-Excite Improves the Tradeoff Between Quality and Training Speed
>
> Squeeze-Excite slows down training, but it leads to quality improvements that make this a worthwhile tradeoff.
> It slows down training because it adds extra computation to the model that decreases the throughput of training.
> However, the training slowdown is a worthwhile tradeoff for the accuracy improvements that Squeeze-Excite produces.

> ❗ Squeeze-Excite Slows Down Inference
>
> Squeeze-Excite adds extra computation to the model that decreases the throughput of inference.
> The inference slowdown will have to be weighed against the benefits of (1) better tradeoffs between training cost and model quality and (2) overall higher attainable model quality.
> This decision must be made in the context of the specific inference workload and the relative value of more efficient training vs. slower inference.

Because SE modules slow down the model, they compose well with methods that make the data loader slower (e.g., RandAugment) or that speed up each training step (e.g., Selective Backprop). In the former case, the slower model allows more time for the data loader to run. In the latter case, the initial slowdown allows techniques that accelerate the forward and backward passes to have a greater effect before they become limited by the data loader's speed.

>  ✅ Squeeze-Excite Can Mitigate Other Bottlenecks
>
> Since Squeeze-Excite decreases GPU throughput, it can reduce relative load on the CPU and data loading pipeline, potentially allowing more CPU-intensive speedup methods (e.g., RandAugment) to run without bottlenecking training on the CPU.

## Attribution

[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) by Jie Hu, Li Shen, and Gang Sun. Published at CVPR 2018.

*This Composer implementation of this method and the accompanying documentation were produced by Davis Blalock and Ajay Saini at MosaicML.*
