# 🫀 Squeeze-and-Excitation

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

Adds a channel-wise attention operator in CNNs. Attention coefficients are produced by a small, trainable MLP that uses the channels' globally pooled activations as input.

| ![Squeeze-Excite](https://storage.googleapis.com/docs.mosaicml.com/images/methods/squeeze-and-excitation.png) |
|:--:
|*After an activation tensor $\mathbf{X}$ is passed through Conv2d $\mathbf{F}_{tr}$ to yield a new tensor $\mathbf{U}$, a Squeeze-and-Excitation (SE) module scales the channels in a data-dependent manner. The scales are produced by a single-hidden-layer fully-connected network whose input is the global-averaged-pooled $\mathbf{U}$. This can be seen as a channel-wise attention mechanism.*|

## How to Use

### Functional Interface

```python
# Run the Blurpool algorithm directly on the model using the Composer functional API

import composer.functional as cf
import torch
import torch.nn.functional as F

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())

    # only need to pass in opt if apply_squeeze_excite is used after optimizer
    # creation; otherwise only the model needs to be passed in
    cf.apply_squeeze_excite(model,
                            optimizers=opt,
                            min_channels=128,
                            latent_channels=64)

    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(10):
        for X, y in train_loader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

```python
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate point in the training loop

from composer.algorithms import SqueezeExcite
from composer.trainer import Trainer

def train_model(model, train_dataloader):
    blurpool = SqueezeExcite(min_channels=128, latent_channels=64)
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      max_duration='10ep',
                      algorithms=[blurpool])
    trainer.fit()
```

## Implementation Details

Squeeze-Excitation blocks apply channel-wise attention to an activation tensor $\mathbf{X}$. The attention coefficients are produced by a single-hidden-layer MLP (i.e., fully-connected network). This network takes in the result of global average pooling $\mathbf{X}$ as its input vector. In short, the average activations within each channel are used to produce scalar multipliers for each channel.

In order to be architecture-agnostic, our implementation applies the SE attention mechanism after individual conv2d modules, rather than at particular points in particular networks. This results in more SE modules being present than in the original paper.

Our implementation also allows applying the SE module after only certain conv2d modules, based on their channel count (see hyperparameter discussion).


## Suggested Hyperparameters

Squeeze Excite has two hyperparameters:

- `latent_channels` - Number of channels to use in the hidden layer of MLP that computes channel attention coefficients.
- `min_channels` - The minimum number of output channels in a Conv2d for an SE module to be added afterward.

We recommend setting `latent_channels` to a value such that the minimimum channel count in any layer will be at least 64. One can accomplish this either by 1) directly setting `latent_channels=64` or more, or 2) by setting `latent_channels=r` and `min_channels=int(64/r)` or more.

We recommend setting `min_channels` to the largest output channel count of any Conv2d in the network.

## Technical Details

This method tends to consistently improve the accuracy of CNNs both in absolute terms and when controlling for training and inference time. This may come at the cost of a roughly 20% increase in inference latency, depending on the architecture and inference hardware.

> ❗ Label Smoothing May Interact with Other Methods that Modify Targets
> Squeeze-Excite will slow down the training and inference, though it tends to add enough accuracy to compensate for this.

Because SE modules slow down the model, they compose well with methods that make the data loader slower (e.g., RandAugment) or that speed up each training step (e.g., Selective Backprop). In the former case, the slower model allows more time for the data loader to run. In the latter case, the initial slowdown allows techniques that accelerate the forward and backward passes to have a greater effect before they become limited by the data loader's speed.

## Attribution

[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) by Jie Hu, Li Shen, and Gang Sun. Published at CVPR 2018.

*This Composer implementation of this method and the accompanying documentation were produced by Davis Blalock and Ajay Saini at MosaicML.*
