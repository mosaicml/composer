# 📎 Gradient Clipping

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Attribution\]](#attribution)

 `Computer Vision`, `Natural Language Processing`

Gradient Clipping is a technique used to stabilize the training of neural networks. It was
originally invented to solve the problem of vanishing and exploding gradients in [training](https://www.fit.vut.cz/study/phd-thesis/283/.en) [recurrent neural networks](https://arxiv.org/abs/1211.5063), but it has also shown to be useful for [transformers](https://arxiv.org/abs/1909.05858v2) and [convolutional](https://arxiv.org/abs/1512.00567v3) [neural networks](https://arxiv.org/abs/2102.06171).
Gradient clipping usually consists of clipping the extreme values of a model's gradients (or the gradients' norms) to be under
a certain threshold. The gradient clipping operation is executed after gradients are computed (after `loss.backward()`), but before the weights of the network are updated (`optim.step()`).

<!--| |
|:--:
|*Need a picture.*|-->

## How to Use
The desired gradient clipping type can be controlled using the `clipping_type` argument.
### The Different Flavors of Gradient Clipping

#### **Gradient clipping by value:**
Constrains all gradients to be between $[-\lambda, \lambda]$, where $\lambda$ is
the `clipping_threshold`.

<!-- Usage: -->
<!--
```python
from tests.common import SimpleModel

model = SimpleModel()
clipping_threshold = 0.1
```
-->
<!--pytest-codeblocks:cont-->
```python
import composer.functional as cf
cf.apply_gradient_clipping(model.parameters(),
                           clipping_type='value',
                           clipping_threshold=clipping_threshold)
```
#### **Gradient clipping by norm:**
Multiplies all gradients by $\min(1, \frac{\lambda}{||G||})$, where $\lambda$ is
the `clipping_threshold` and $||G||$ is the total L2 norm of all gradients.
<!-- Usage: -->
<!--
```python
from tests.common import SimpleModel

model = SimpleModel()
clipping_threshold = 0.1
```
-->
<!--pytest-codeblocks:cont-->
```python
import composer.functional as cf
cf.apply_gradient_clipping(model.parameters(),
                           clipping_type='norm',
                           clipping_threshold=clipping_threshold)
```
#### **Adaptive Gradient Clipping (AGC):**
Clips all gradients based on the gradient norm to parameter norm ratio by multiplying them by
$\min(1, \lambda\frac{||W||}{||G||})$, where $\lambda$ is the `clipping_threshold`,
$||G||$ is the norm of the gradients and $||W||$ is the norm of the weights.
<!-- Usage: -->
<!--
```python
from tests.common import SimpleModel

model = SimpleModel()
clipping_threshold = 0.1
```
-->
<!--pytest-codeblocks:cont-->
```python
import composer.functional as cf
cf.apply_gradient_clipping(model.parameters(),
                           clipping_type='adaptive',
                           clipping_threshold=clipping_threshold)
```

### Functional Interface

```python
# Run gradient clipping directly on the model right after a loss.backward() call
# using the Composer functional API.

import torch
import composer.functional as cf

clipping_type = 'norm' # can also be 'adaptive' or 'value'
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
            cf.apply_gradient_clipping(model.parameters(), clipping_type=clipping_type,
                                        clipping_threshold=0.1)
            opt.step()
```

### Composer Trainer

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
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate points in the training loop

from composer.algorithms import GradientClipping
from composer.trainer import Trainer

clipping_type = 'norm' # can also be 'adaptive' or 'value'
gc = GradientClipping(clipping_type=clipping_type, clipping_threshold=0.1)

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep',
    algorithms=[gc]
)

trainer.fit()
```

### Implementation Details

#### `clipping_type='norm'`
Norm-based gradient clipping is implemented as follows:
On `Event.AFTER_TRAIN_BATCH`, for every parameter in the model that has gradients:
1. Compute the parameter's gradients and concatenate all parameters' gradients into one big vector
2. Compute the norm of all the gradients (single scalar), $||G||$
3. Compute the clipping coefficient, `clip_coeff`: $\lambda / ||G||$
4. Clamp the `clip_coeff` to be less than or equal to 1.0
5. Multiply all the gradients by the `clip_coeff`.

#### `clipping_type='value'`
Value-based gradient clipping is implemented as follows:
On `Event.AFTER_TRAIN_BATCH`, for every parameter in the model that has gradients:
1. Any gradients that are greater than `clipping_threshold` are set to `clipping_threshold` and
any gradients less than -`clipping_threshold` are set to -`clipping_threshold`. See [here](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html) for more details.
#### `clipping_type='adaptive'`

Adaptive gradient clipping is implemented as follows:

On `Event.AFTER_TRAIN_BATCH`, for every parameter in the model that has gradients:
1. Compute the parameter's weight norm with an L2 norm (normalized across rows for MLP's, across entire filters for CNN's, and across the entire vector for biases), $||W||$
2. Compute the parameter's gradient norm with an L2 norm, $||G||$
3. If $||G|| > \lambda||W||$, scale all the contributing gradients by $\lambda \frac{||W||}{||G||}$.


## Suggested Hyperparameters
### Norm-based gradient clipping
The [original authors, R. Pascanu](https://arxiv.org/abs/1211.5063) of this type of clipping used gradient clipping with recurrent neural networks. They recommend monitoring the average gradient norm of your model's gradients over many iterations as a heuristic to help
figure out a value for the `clipping_threshold`.

For computer vision, the authors of the famous [Inception convolutional neural network architecture](https://arxiv.org/abs/1512.00567v3) used a `clipping_threshold` of 2.0, which they claim helped stabilize their training.

For NLP with transformers, [Keskar, et al](https://arxiv.org/abs/1909.05858v2) used a `clipping_threshold` of 0.25 for their CTRL, a conditional transformer language model.
The [authors of TABERT](https://arxiv.org/abs/2005.08314v1), a transformer-based BERT model tabular data, recommend a `clipping_threshold` of 1.0. The [authors of the Compressive Transformer](https://arxiv.org/abs/1911.05507v1) and [Gated Convolutional Neural Networks](https://arxiv.org/pdf/1612.08083v3.pdf) both used a `clipping_threshold` of 0.1.

### Value-based gradient clipping
The [original author of this type of clipping, Mikolov](https://www.fit.vut.cz/study/phd-thesis/283/.en) uses it for training recurrent neural networks and recommends setting the `clipping_threshold` to 15. This approach to gradient clipping is not as prevalent as the norm-based clipping and thus to our knowledge there are not very many examples of good settings for `clipping_threshold`.
### Adaptive gradient clipping
We haven't done much experimentation with AGC. However, [the original authors, Brock et al.](https://arxiv.org/abs/2102.06171)
and [Ayush Thakur](https://wandb.ai/ayush-thakur/nfnet/reports/Exploring-Adaptive-Gradient-Clipping-and-NFNets--Vmlldzo1MDc0NTQ)
have done some ablations have some recommendations. Note, both parties use AGC with NF-ResNets, which is a variation
of ResNets that removes Batch Norm and includes [Weight Standardization](https://arxiv.org/abs/1903.10520)
among other modifications.

Brock et al. recommend using a `clipping threshold` of 0.01 for batch sizes between 1024 to 4096.
For smaller batch sizes, AGC's effects are less pronounced they recommend a larger (less strict) `clipping factor` with performance
slightly increasing up to 0.08. They also recommend removing AGC from the last linear layer of the network.

Thakur recommends large `clipping threshold` for small batch sizes (at least 0.16 for batch sizes 128 and 256) and smaller `clipping threshold` for large batch sizes. They also found that AGC seems to work especially well for the NF-ResNet architecture. Specifically they found that for `clipping threshold` of 0.01 and batch size of 1024, AGC does not improve the the performance of a vanilla ResNet with Batch Norm removed.

<!-- ## Technical Details
TODO(eracah): fill in this section.
-->


## Attribution

[*High-Performance Large-Scale Image Recognition Without Normalization*](https://arxiv.org/abs/2102.06171) by Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan. Published in ICML 2021.

[*On the difficulty of training recurrent neural networks*](https://arxiv.org/abs/1211.5063) by R. Pascanu, T. Mikolov, and Y. Bengio, 2012

[*Statistical language models based on neural networks*](https://www.fit.vut.cz/study/phd-thesis/283/.en) by T. Mikolov

*The Composer implementation of this method and the accompanying documentation were produced by Evan Racah at MosaicML.*
