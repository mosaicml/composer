# ⛩️ Gated Linear Units

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution) - [\[API Reference\]](#api-reference)

 `Natural Language Processing`

Gated Linear Units replaces the projection matricies in the feed-forward block with [Gated Linear Units](https://arxiv.org/abs/2002.05202).

| ![GatedLinearUnits](https://storage.cloud.google.com/docs.mosaicml.com/images/methods/gated_linear_units.png)|
|:--|
|*These equations compare the projection matricies in a standard feed-forward network, and a Gated Linear Unit.
Following [Shazeer, 2020](https://arxiv.org/abs/2002.05202), we omit the use of bias terms. $\cdot$ represents a dot product.*|

## How to Use

### Functional Interface

```python
# Apply surgery on the model to swap the feed-forward block
# for a gated feed-forward block using the Composer Functional API

import composer.functional as cf

def training_loop(model, train_loader):
    cf.apply_gated_linear_units(model)

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
from composer.algorithms import GatedLinearUnits

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  max_duration='1ep',
                  algorithms=[GatedLinearUnits()])

trainer.fit()
```

### Implementation Details

Gated Linear Units provide a more expressive form for a feed-forward block by performing a "gating" operation on the input matrix. The careful reader will recognize that we introduce a new weight matrix, $W_3$. In order to iso-parameter experiments, we scale $D_{ff}$ by $\frac{2}{3}$.
This algorithm significant improves convergence, but with a slight degredation to throughput. We recommend training with `bias = False`, even if biases are enabled in the rest of your model. This substantially improved throughput and convergence.

## Suggested Hyperparameters

While hyperparameters can vary significantly per use case, we recommend training with
```
act_fn = {ReLU, GeLU},
gated_layer_bias = False,
non_gated_layer_bias = False
```
We observed that, on average, GeLU activation functions marginally performed better than ReLU activation functions, and observed a significant improvement from using GeLU and ReLU over Swish and a Squared ReLU. We observed a significant benefit from setting `bias = False` for both weight matricies $W_1$ and $W_3$.

## Technical Details

> ✅ Gated Linear Units Improve the Tradeoff Between Quality and Training Speed

While there are many hypotheses for the performace of Gated Linear Units, the community lacks a through investigation of these. The algorithm has been shown to perform well empirically, and there remains an open curiosity as to why step-wise convergence is significantly better without bias terms than with bias terms. Furthermore, in order to maximize throughput, the user should make sure that the scaled down feature dimension when using GLUs is still a multiple of eight.

## Attribution

[*Language Modeling with Gated Convolutional Networks*](https://arxiv.org/abs/1612.08083)
by Dauhpin et al. Posted to arXiv in 2016.

Our implementation most closely matches: [*GLU Variants Improve
Transformer*](https://arxiv.org/abs/2002.05202) by Shazeer. Posted to arXiv in 2020.

*The Composer implementation of this method and the accompanying documentation were produced by Moin Nadeem at MosaicML.*

## API Reference

**Algorithm class:** {class}`composer.algorithms.GatedLinearUnits`

**Functional:** {func}`composer.functional.apply_gated_linear_units`
