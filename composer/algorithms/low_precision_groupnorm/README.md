# 🧁 Low Precision GroupNorm


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Natural Language Processing`, `Math Equivalent`

Low Precision GroupNorm forces `torch.nn.GroupNorm` modules to run in float16 or bfloat16 precision, improving utilization. This should not affect final convergence values, but in rare cases may cause loss spikes.


## How to Use
Low Precision GroupNorm wraps `torch.nn.GroupNorm`, forcing the module to run in a lower precision if you have autocast enabled. If you are running in Automatic Mixed Precision (`amp`) mode, Low Precision GroupNorm will run in `torch.float16`. If you are running in `bf16` mode, Low Precision GroupNorm will run in `torch.bfloat16`.

This algorithm will have no effect if you are running in `fp32` or `fp16` mode.


### Functional Interface

```python
# Apply surgery on the model to swap-in the Low Precision GroupNorm using the Composer functional API

import composer.functional as cf

def training_loop(model, train_loader):
    cf.apply_low_precision_groupnorm(model, precision='amp')

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
<!--
```python
from tests.common.models import SimpleGroupNormConvModel
from torch.utils.data import DataLoader
from tests.common import RandomImageDataset

model = SimpleGroupNormConvModel()
train_dataloader = DataLoader(RandomImageDataset(), batch_size=2)
eval_dataloader = DataLoader(RandomImageDataset(), batch_size=2)
```
-->
<!--pytest-codeblocks:cont-->
```python
from composer.trainer import Trainer
from composer.algorithms import LowPrecisionGroupNorm

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='1ep',
                  algorithms=[LowPrecisionGroupNorm()])

trainer.fit()
```

### Implementation Details

Low Precision GroupNorm is implemented by performing model surgery, which looks for instances of `torch.nn.GroupNorm` and replaces them with `composer.algorithms.low_precision_groupnorm.low_precision_groupnorm.LPGroupNorm`, which is a thin wrapper around `torch.nn.GroupNorm` that manually turns autocast off and specifices the input dtype to lower precision.

## Suggested Hyperparameters

Low Precision GroupNorm utilizes the existing parameters from the original model. The functional version of Low Precision GroupNorm allows users to specify the `precision` mode, which should be set to the Composer precision format of your model. When using the algorithm through the Composer trainer, Low Precision GroupNorm will use the trainer's `precision` mode.

## Technical Details

Low Precision GroupNorm is meant to replace our Fused GroupNorm algorithm. The two algorithms achieve very similar throughput. Fused GroupNorm also runs in low precision, but it is a more complex algorithm, since it uses a custom kernel. Since the custom kernel provides no additional speedup, we have replaced it with this simpler algorithm.

> ✅ Low Precision GroupNorm Improves Training Speed
>
> In our experiments, Low Preicision GroupNorm improves the attainable tradeoffs between training speed and the final quality of the trained model.
> We recommend using Low Precision GroupNorm.

## Attribution

*The Composer implementation of this method and the accompanying documentation were produced by MosaicML.*

## API Reference

**Algorithm class:** {class}`composer.algorithms.LowPrecisionGroupNorm`

**Functional:** {func}`composer.functional.apply_low_precision_groupnorm`
