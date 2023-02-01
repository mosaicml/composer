# 🧁 Low Precision LayerNorm


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Natural Language Processing`, `Math Equivalent`

Low Precision LayerNorm forces `torch.nn.LayerNorm` modules to run in float16 or bfloat16 precision, improving utilization. This should not affect final convergence values, but in rare cases may cause loss spikes. This algorithm is an alternative to Fused LayerNorm, and should give very similar performance.


## How to Use
Low Precision LayerNorm wraps `torch.nn.LayerNorm`, forcing the module to run in a lower precision if you have autocast enabled. If you are running in Automatic Mixed Precision (`amp`) mode, Low Precision LayerNorm will run in `torch.float16`. If you are running in `bf16` mode, Low Precision LayerNorm will run in `torch.bfloat16`.

This algorithm will have no effect if you are running in `fp32` or `fp16` mode.


### Functional Interface

```python
# Apply surgery on the model to swap-in the Low Precision LayerNorm using the Composer functional API

import composer.functional as cf

def training_loop(model, train_loader):
    cf.apply_low_precision_layernorm(model, precision='amp')

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
from tests.common.models import configure_tiny_bert_hf_model
from tests.common.datasets import dummy_bert_lm_dataloader

model = configure_tiny_bert_hf_model()
train_dataloader, eval_dataloader = dummy_bert_lm_dataloader(), dummy_bert_lm_dataloader()
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

Low Precision LayerNorm is implemented by performing model surgery, which looks for instances of `torch.nn.LayerNorm` and replaces them with `composer.algorithms.low_precision_layernorm.low_precision_layernorm.LPLayerNorm`, which is a thin wrapper around `torch.nn.LayerNorm` that manually turns autocast off and specifices the input dtype to lower precision. In `bf16` mode on PyTorch versions prior to 1.13, Low Precision LayerNorm will fall back to Fused LayerNorm, replacing instances of `torch.nn.LayerNorm` with `apex.normalization.fused_layer_norm`.

## Suggested Hyperparameters

Low Precision LayerNorm utilizes the existing `normalized_shape` and `d_eps` from the original model. The functional version of Low Precision LayerNorm allows users to specify the `precision` mode, which should be set to the Composer precision format of your model. When using the algorithm through the Composer trainer, Low Precision LayerNorm will use the trainer's `precision` mode.

## Technical Details

Low Precision LayerNorm is meant to replace our Fused LayerNorm algorithm. The two algorithms achieve very similar throughput. Fused LayerNorm also runs in low precision, but it is a more complex algorithm, since it uses a custom kernel. Since the custom kernel provides no additional speedup, we have replaced it with this simpler algorithm.

> ✅ Low Precision LayerNorm Improves Training Speed
>
> In our experiments, Low Preicision LayerNorm improves the attainable tradeoffs between training speed and the final quality of the trained model.
> We recommend using Low Precision LayerNorm.

## Attribution

*The Composer implementation of this method and the accompanying documentation were produced by MosaicML.*

## API Reference

**Algorithm class:** {class}`composer.algorithms.LowPrecisionLayerNorm`

**Functional:** {func}`composer.functional.apply_low_precision_layernorm`
