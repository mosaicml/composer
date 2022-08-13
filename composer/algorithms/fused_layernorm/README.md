# 🍰 Fused LayerNorm


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Natural Language Processing`

Fused LayerNorm replaces implementations of `torch.nn.LayerNorm` with a `apex.normalization.fused_layer_norm`. The fused kernel provides increased GPU utilization.

| ![FusedLayerNorm](https://miro.medium.com/max/1200/0*ugfR_r4J9PK8tXNb)|
|:--|
|*A visualization of the impact of Fused LayerNorm.*|

## How to Use

### Functional Interface

```python
# Apply surgery on the model to swap-in the Fused LayerNorm using the Composer functional API

import composer.functional as cf

def training_loop(model, train_loader):
    cf.apply_fused_layernorm(model)

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
from composer.algorithms import FusedLayerNorm

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='1ep',
                  algorithms=[FusedLayerNorm()])

trainer.fit()
```

### Implementation Details

Fused LayerNorm is implemented by performing model surgery, which looks for instances of `torch.nn.LayerNorm` and replaces them with a `apex.normalization.fused_layer_norm`. This should be applicable to any model that utilizes a `torch.nn.LayerNorm`.

## Suggested Hyperparameters

Fused LayerNorm does not have any hyperparameters. It utilizes the existing `normalized_shape` and `d_eps` from the original model.

## Technical Details

APEX's FusedLayerNorm achieves a substantial speedup over PyTorch by doing a few things:
1. Instead of a naive implementation, which requires two passes over the input in order to estimate variances, it uses [Welford's Online Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm) to estimate the variances in a single step, creating a substantive wall-clock speedup.
2. Instead of requiring multiple CUDA kernel launches, it computes everything in a single kernel launch, therefore improving GPU utilization.

## Attribution

*The Composer implementation of this method and the accompanying documentation were produced by Moin Nadeem at MosaicML.*
