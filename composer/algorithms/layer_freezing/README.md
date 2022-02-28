# ❄️ Layer Freezing


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Computer Vision`

Layer Freezing gradually makes early modules untrainable ("freezing" them), saving the cost of backpropagating to and updating frozen modules.
The hypothesis behind Layer Freezing is that early layers may learn their features sooner than later layers, meaning they do not need to be updated as late into training.

<!--| ![LayerFreezing](https://storage.googleapis.com/docs.mosaicml.com/images/methods/layer-freezing.png) |
|:--:
|*Need a picture.*|-->

## How to Use

<!--### Functional Interface

TODO(CORY): FIX

```python
def training_loop(model, train_loader):
  opt = torch.optim.Adam(model.parameters())
  loss_fn = F.cross_entropy
  model.train()
  
  for epoch in range(num_epochs):
      for X, y in train_loader:
          y_hat = model(X)
          loss = loss_fn(y_hat, y)
          loss.backward()
          opt.step()
          opt.zero_grad()
```

### Composer Trainer

TODO(CORY): Verify and provide commentary and/or comments

```python
from composer.algorithms import XXX
from composer.trainer import Trainer

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  max_duration='1ep',
                  algorithms=[
                  ])

trainer.fit()
```-->

### Implementation Details

At the end of each epoch after `freeze_start`, the algorithm traverses the ownership tree of `torch.nn.Module` objects within one’s model in depth-first order to obtain a list of all modules. Note that this ordering may differ from the order in which modules are actually used in the forward pass.
Given this list of modules, the algorithm computes how many modules to freeze. This number increases linearly over time such that no modules are frozen at `freeze_start` and a fraction equal to `freeze_level` are frozen at the end of training.
Modules are frozen by removing their parameters from the optimizer’s `param_groups`. However, their associated state dict entries are not removed.

## Suggested Hyperparameters

Layer Freezing works best when the entire network is trainable before freezing begins.
We have found that `freeze_start` should be at least `0.1`.
The setting of `freeze_level` is context specific. TODO(CORY): Say what we used for ResNet-50 on ImageNet in the explorer runs.

## Technical Details

Layer freezing begins freezing the earliest layers in the network at the point in training specified by `freeze_start` (e.g., 10% of the way into training if `freeze_start=0.1`).
At that point, it begins freezing modules early in the network.
Over the remainder of training, it progressively freezes later layers in the network.
It freezes these layers linearly over time until the latest layer to be frozen (specified by `freeze_level`) gets frozen prior to the end of training.

We have yet to observe a significant improvement in the tradeoff between speed and accuracy using this Layer Freezing on our computer vision benchmarks.
We’ve observed that layer freezing can increase throughput by ~5% for ResNet-50 on ImageNet, but decreases accuracy by 0.5-1%. This is not an especially good speed vs accuracy tradeoff. Existing papers have generally also not found effective tradeoffs on large-scale problems.
For ResNet-56 on CIFAR-100, we have observed an accuracy lift from 75.82% to 76.22% with a similar ~5% speed increase. However, these results used specific hyperparameters without replicates, and so should be interpreted with caution.


> ❗ There is No Evidence that Layer Freezing Improves the Tradeoff Between Model Quality and Training Speed
>
>  Although layer freezing does improve throughput, it also leads to accuracy reductions we observed on ResNet-50 on ImageNet.
>  This tradeoff between improved throughput and reduced quality was not worthwhile in our experiments: it did not improve the pareto frontier of the tradoeff between quality and training speed.


> 🚧 Composing Regularization Methods
>
> Layer freezing is a relaxed version of early stopping that stops training the model gradually, rather than all at once. It can therefore be understood as a form of regularization. As general rule, composing regularization methods may lead to diminishing returns in quality improvements.

## Attribution

Freezing layers is an old and common practice, but our precise freezing scheme most closely resembles [*Freezeout: Accelerate training by progressively freezing layers*](https://arxiv.org/abs/1706.04983) by Andrew Brock, Theodore Lim, J.M. Ritchie, and Nick Westin (posted on arXiv in 2017) and [*SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability*](https://arxiv.org/abs/1706.05806) by Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein (published in NeurIPS 2017).

*The Composer implementation of this method and the accompanying documentation were produced by Cory Stephenson at MosaicML.*
