# 🍰 Low Precision LayerNorm


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Natural Language Processing`

Low Precision LayerNorm forces `torch.nn.LayerNorm` modules to run in float16 or bfloat16 precision, improving utilization. This should not affect final convergence values, but may result in occassional loss spikes.


## How to Use

### Functional Interface

```python
# Apply surgery on the model to swap-in the Low Precision LayerNorm using the Composer functional API

import composer.functional as cf

def training_loop(model, train_loader):
    cf.apply_low_precision_layernorm(model)

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
from tests.fixtures.synthetic_hf_state import make_dataset_configs, synthetic_hf_state_maker

synthetic_config = make_dataset_configs(model_family=['bert'])[0]
_, model, train_dataloader = synthetic_hf_state_maker(synthetic_config)
_, _, eval_dataloader = synthetic_hf_state_maker(synthetic_config)
```
-->
<!--pytest-codeblocks:cont-->
```python
from composer.trainer import Trainer
from composer.algorithms import LowPrecisionLayerNorm

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='1ep',
                  algorithms=[LowPrecisionLayerNorm()])

trainer.fit()
```

### Implementation Details

Low Precision LayerNorm is implemented by performing model surgery, which looks for instances of `torch.nn.LayerNorm` and manually turns autocast off and specifices the input dtype.

## Suggested Hyperparameters

Low Precision LayerNorm does not have any hyperparameters. It utilizes the existing `normalized_shape` and `d_eps` from the original model.

## Technical Details

Low Precision LayerNorm replaces our implementation of APEX's Fused LayerNorm. The two algorithms achieve very simila throughput. Fused LayerNorm also ran in low precision mode, but it is a more complex algorithm, since it replaces the `torch.nn.LayerNorm` module with a custom kernel. Since the custom kernel provides no additional speedup, we have replaced it with this simpler algorithm.

## Attribution

*The Composer implementation of this method and the accompanying documentation were produced by MosaicML.*
