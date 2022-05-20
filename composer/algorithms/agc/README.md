# 📎 AGC

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Computer Vision`

AGC (Adaptive Gradient Clipping) .

<!--| ![AGC](https://storage.googleapis.com/docs.mosaicml.com/images/methods/agc.png) |
|:--:
|*Need a picture.*|-->

## How to Use

### Functional Interface

```python
# Run the AGC algorithm directly on the model right after a loss.backward() call
# using the Composer functional API.

import torch
import composer.functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()
  
    for epoch in range(num_epochs):
        for X, y in train_loader:
            opt.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            cf.apply_agc(model)
            opt.step()
```

### Composer Trainer

<!-- TODO: Address timeouts -->
<!--pytest-codeblocks:skip-->
```python
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate points in the training loop

from composer.algorithms import AGC
from composer.trainer import Trainer

agc = AGC(clipping_threshold = 0.01)

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    max_duration='1ep',
    algorithms=[agc]
)

trainer.fit()
```

### Implementation Details

AGC is implemented as follows:

On `Event.AFTER_TRAIN_BATCH`, for every parameter in the model that has gradients:
1. Compute the parameter's weight norm with an L2 norm (normalized across rows for MLP's, across entire filters for CNN's, and across the entire vector for biases).
2. Compute the parameter's gradient norm with an L2 norm.
3. If `grad_norm > weight_norm * clipping_threshold`, scale all the contributing gradients by `clipping_threshold * (weight_norm / grad_norm)`. 


## Suggested Hyperparameters

We haven't done much experimentation with AGC. However, [the original authors, Brock et al.](https://arxiv.org/abs/2102.06171)
and [Ayush Thakur](https://wandb.ai/ayush-thakur/nfnet/reports/Exploring-Adaptive-Gradient-Clipping-and-NFNets--Vmlldzo1MDc0NTQ)
have done some ablations have some recommendations. Note, both parties use AGC with NF-ResNets, which is a variation
of ResNets that removes Batch Norm and includes [Weight Standardization](https://arxiv.org/abs/1903.10520) 
among other modifications.

Brock et al. recommend using a `clipping threshold` of 0.01 for batch sizes between 1024 to 4096.
For smaller batch sizes, AGC's effects are less pronounced they recommend a larger (less strict) `clipping factor` with performance
slightly increasing up to 0.08. They also recommend removing AGC from the last linear layer of the network.

Thakur recommends large `clipping threshold` for small batch sizes (at least 0.16 for batch sizes 128 and 256) and smaller `clipping threshold` for large batch sizes .
They also found that AGC seems to work especially well for the NF-ResNet architecture. Specifically they found that for `clipping threshold` of 0.01 and batch size of 1024, AGC does not improve the the performance of a vanilla ResNet with Batch Norm removed.

<!-- ## Technical Details 
TODO(eracah): fill in this section.
-->


## Attribution

[*High-Performance Large-Scale Image Recognition Without Normalization*](https://arxiv.org/abs/2102.06171) by Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan. Published in ICML 2021.

*The Composer implementation of this method and the accompanying documentation were produced by Evan Racah at MosaicML.*
