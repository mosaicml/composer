# 🧩 Stochastic Weight Averaging

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Attribution\]](#attribution)

 `Computer Vision`, `Natural Language Processing`

Stochastic Weight Averaging (SWA) maintains a running average of the weights towards the end of training. This leads to better generalization than conventional training.

| ![SWA](https://storage.googleapis.com/docs.mosaicml.com/images/methods/swa.png) |
|:--|
|*Visualization of SWA from an extensive [PyTorch blogpost about SWA](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/).*|


## How to Use

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

from composer.algorithms import SWA
from composer.trainer import Trainer

swa_algorithm = SWA(
    swa_start="1ep",
    swa_end="2ep",
    update_interval='5ba',
)
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration="3ep",
    algorithms=[swa_algorithm]
)

trainer.fit()
```

## Implementation Details

SWA stores an extra copy of the model's (averaged) weights, and thus doubles the memory required for the model.

EMA also uses a small amount of compute to calculate the average. This does not have an
appreciable impact on training speed unless the averaged model is being updated very
frequently (e.g. ≤ every ten batches).

## Suggested Hyperparameters

- `swa_start`: proportion of training completed before SWA is applied. The
default value is `'0.7dur'` (0.7 of the duration of training).
- `swa_end`: proportion of training completed before SWA ends. It's important to have at
  least one epoch of training after the baseline model is replaced by the SWA model so
  that the SWA model can have its buffers (most importantly its batch norm statistics)
  updated. If ``swa_end`` occurs during the final epoch of training (e.g. ``swa_end =
  0.9dur`` and training is only 5 epochs, or ``swa_end = 1.0dur``), the SWA model will not
  have its buffers updated, which can negatively impact accuracy. The default value is ``'0.97dur'``.
- `update_interval` - The period at which updates to the moving average are computed. A
  longer update interval means that updates are computed less frequently.

Our implementation of SWA also has hyperparameters to control an SWA-specific learning
rate schedule, but we found that these did not have a substantial impact on training.

## Attribution

[*Averaging Weights Leads to Wider Optima and Better Generalization*](https://arxiv.org/abs/1803.05407) by Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Presented at the 2018 Conference on Uncertainty in Artificial Intelligence.
