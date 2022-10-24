```{eval-rst}
:orphan:
```

# 🏋️‍♀️ Decoupled Weight Decay

[\[How to Use\]](#how-to-use) - [\[Suggested
Hyperparameters\]](#suggested-hyperparameters) - [\[Technical
Details\]](#technical-details) - [\[Attribution\]](#attribution) - [\[API Reference\]](#api-reference)

L2 regularization is typically considered equivalent to weight decay, but this equivalence only holds for certain optimizer implementations. Common optimizer implementations typically scale the weight decay by the learning rate, which complicates model tuning and hyperparameter sweeps by coupling the learning rate and weight decay. Implementing weight decay explicitly and separately from L2 regularization allows for a new means of tuning regularization in models.

## How to Use
### Composer Trainer
<!--pytest.mark.gpu-->
<!--
```python
from torch.utils.data import DataLoader
from tests.common import RandomImageDataset

from composer.models import composer_resnet

model = composer_resnet('resnet50')

train_dataloader = DataLoader(RandomImageDataset(), batch_size=2)
eval_dataloader = DataLoader(RandomImageDataset(), batch_size=2)
```
-->
<!--pytest-codeblocks:cont-->
```python
# Instantiate the optimizer and pass it into the Trainer

from composer.optim import DecoupledSGDW
from composer.trainer import Trainer

optimizer = DecoupledSGDW(
    model.parameters(),
    lr=0.05,
    momentum=0.9,
    weight_decay=2.0e-3
)

trainer = Trainer(model=model,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_dataloader,
                    max_duration='1ep',
                    optimizers=optimizer)

trainer.fit()
```

### Implementation Details
Unlike most of our other methods, we do not implement decoupled weight decay as an algorithm, instead providing optimizers that can be used as drop-in replacements for `torch.optim.SGD` and `torch.optim.Adam`; though note that some hyperparameter tuning may be required to realize full performance improvements.

The informed reader may note that PyTorch already provides a `torch.optim.AdamW` variant that implements Loshchilov et al.'s method. Unfortunately, this implementation has a fundamental bug owing to PyTorch's method of handling learning rate scheduling. In this implementation, learning rate schedulers attempt to schedule the weight decay (as Loshchilov et al. suggest) by tying it to the learning rate. However, this means that weight decay is now implicitly tied to the initial learning rate, resulting in unexpected behavior where runs with different learning rates also have different effective weight decays. See [this line](https://github.com/pytorch/pytorch/blob/d921891f5788b37ea92eceddf7417d11e44290e6/torch/optim/_functional.py#L125).

## Suggested Hyperparameters

Optimizers with decoupled weight decay can be used as drop-in replacements for their
non-decoupled counterparts. However, the optimal `weight_decay` value for decoupled
optimizers will typically be smaller than for their non-decoupled counterparts, because
decoupled weight decay is not scaled by the learning rate.
## Technical Details

There are no known negative side effects to using decoupled weight decay once it is properly tuned, as long as the original base optimizer is either `torch.optim.Adam` or `torch.optim.SGD`.

Weight decay is a regularization technique and thus is expected to yield diminishing returns when composed with other regularization techniques.

## Attribution

*[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)*, by Ilya
Loshchilov and Frank Hutter. Published as a conference paper at ICLR 2019.

## API Reference

**Optimizer classes:** {class}`composer.optim.DecoupledAdamW`, {class}`composer.optim.DecoupledSGDW`

**Optimizer module:** {mod}`composer.optim`
