# 🚚 EMA

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

Exponential Moving Average (EMA) is a model averaging technique that maintains an exponentially weighted moving average of the model parameters during training. The averaged parameters are used for model evaluation. EMA typically results in less noisy validation metrics over the course of training, and sometimes increased generalization.

## How to Use

### Functional Interface

```python
# Run the EMA algorithm directly on the batch data using the Composer functional API

import copy

import composer.functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    ema_model = copy.deepcopy(model)
    model.train()

    for epoch in range(num_epochs):
        for X, y in train_loader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            cf.compute_ema(model, ema_model, smoothing=0.99)
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

from composer.algorithms import EMA
from composer.trainer import Trainer

ema = EMA(half_life='50ba')

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='1ep',
                  algorithms=[ema])

trainer.fit()

model = ema.ema_model
```

### Implementation Details

Because EMA needs to maintain a copy of the model's (averaged) weights, it requires a bit more on-device memory. In the functional implementation, the amount of extra memory is 2x the size of the model. In the composer trainer implementation, it is 3x the size of the model to allow for swapping the training and evaluation models. In practice, the extra memory used is small relative to the total amount of memory used, as activations and optimizer state are not duplicated.

EMA also uses a bit of extra compute to calculate the moving average. This can lead to a small slowdown. The extra compute can be reduced by not computing the moving average every iteration. In the composer trainer implementation this can be done by using a larger `update_interval`. In practice we find that as long as `half_life` is much larger than `update_interval`, increasing `update_interval` does not have much effect on generalization performance.

## Suggested Hyperparameters

The Composer Trainer implementation of EMA has two hyperparameters:

- `half_life` - The half life for terms in the average. A longer half life means old information is remembered longer, a shorter half life means old information is discared sooner.
- `update_interval` - The period at which updates to the moving average are computed. A longer update interval means that updates are computed less frequently.

A good typical starting value for `half_life` is `half_life="100ba"`, for a half life of 100 batches. At the same time, `update_interval` can be left unspecified which will default to `update_interval="1ba"`, or set to a larger value such as `update_interval="10ba"` to improve runtime. Shorter update intervals typically result in better generalization performance at the cost of somewhat increased runtime.

Our implementation of EMA also provides the option to use the EMA weights as the training weights, which can be enabled by setting `train_with_ema_weights=True`. We reccomend leaving this off with the default value of `train_with_ema_weights=False.`

## Technical Details

>  ✅ EMA should result in less noisy validation metrics during training
>
> If evalutation metrics are computed over the course of training, EMA should result in these metrics being smoother and less noisy due to averaging.

> ❗ EMA increases memory consumption
>
> Because EMA needs to maintain a copy of the model's (averaged) weights, it requires a bit more on device memory. In practice, the extra memory used is small relative to the total amount of memory used, as activations and optimizer state are not duplicated.

> ❗ EMA uses some extra compute
>
>This can lead to a small slowdown. The extra compute can be reduced by not computing the moving average every iteration. In the composer trainer implementation this can be done by using a larger `update_interval`.

> ❗ Evaluation should not be done with the training model
>
> Evaluation should be done with the `ema_model` in the functional impementation as this is the model containing the averaged parameters. The ema model can be accessed after training from the `EMA` object via `model = ema.get_ema_model(model)` in the composer trainer implementation.


## Attribution

Our implementation of EMA was inspired by [Tensorflow's Exponential Moving Average](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)

*This Composer implementation of this method and the accompanying documentation were produced by Cory Stephenson at MosaicML.*
