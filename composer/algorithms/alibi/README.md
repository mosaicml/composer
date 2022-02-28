# 🥸 ALiBi

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Natural Language Processing`

ALiBi (Attention with Linear Biases) dispenses with position embeddings for tokens in transformer-based NLP models, instead encoding position information by biasing the query-key attention scores proportionally to each token pair’s distance. ALiBi yields excellent extrapolation to unseen sequence lengths compared to other position embedding schemes. We leverage this extrapolation capability by training with shorter sequence lengths, which reduces the memory and computation load.

| ![Alibi](https://storage.googleapis.com/docs.mosaicml.com/images/methods/alibi.png) |
|:--:
|*The matrix on the left depicts the attention score for each key-query token pair. The matrix on the right depicts the distance between each query-key token pair. m is a head-specific scalar that is fixed during training. Figure from [Press et al., 2021](https://openreview.net/forum?id=R8sQPpGCv0).*|

## How to Use

### Functional Interface

```python
# Run the ALiBi algorithm directly on the model using the Composer functional API 

import torch
import composer.functional as cf

from composer.algorithms.alibi.gpt2_alibi import _attn
from composer.algorithms.alibi.gpt2_alibi import enlarge_mask
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

def training_loop(model, train_loader):
    cf.apply_alibi(model=model,
                   heads_per_layer=12,
                   max_sequence_length=8192,
                   position_embedding_attribute="module.transformer.wpe",
                   attention_module=GPT2Attention,
                   attr_to_replace="_attn",
                   alibi_attention=_attn,
                   mask_replacement_function=enlarge_mask)

    opt = torch.optim.Adam(model.parameters())
    loss_fn = F.cross_entropy
    model.train()
  
    for epoch in range(num_epochs):
        for X, y in train_loader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

```python
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate points in the training loop

from composer.algorithms import Alibi
from composer.trainer import Trainer

alibi = Alibi(position_embedding_attribute="module.transformer.wpe",
              attention_module_name="transformers.models.gpt2.modeling_gpt2.GPT2Attention",
              attr_to_replace="_attn",
              alibi_attention="composer.algorithms.alibi._gpt2_alibi._attn",
              mask_replacement_function="composer.algorithms.alibi.gpt2_alibi.enlarge_mask")

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  max_duration='1ep',
                  algorithms=[alibi])

trainer.fit()
```

### Implementation Details

ALiBi is implemented as follows. On `Event.INIT`:
1. The model's position embeddings are expanded to accommodate sequences of up to length `max_sequence_length`, and then "bypassed" by setting them to zero and freezing them.
2. The attribute that computes the self-attention (`attr_to_replace`) in the model's self-attention modules (`attention_module_name`) is replaced with an ALiBi-enabled self-attention method (`alibi_attention`) using graph surgery.
On `Event.AFTER_DATALOADER`, the length of training data sequences in a batch are scaled by `train_sequence_length_scaling` by reshaping the data tensors.


## Suggested Hyperparameters

We found that `train_sequence_length_scaling=0.25` (sequence length 256) provided appreciable speed and accuracy gains for models evaluated at sequence length 1024.
We observed that performance significantly degraded for ALiBi models trained on sequence lengths ≤128.
As such, we do not recommend training models with sequence lengths ≤256 or `train_sequence_length_scaling≤0.03125`, whichever is larger.

## Technical Details

ALiBi dispenses with traditional position embeddings and instead adds a static, non-learned bias to the query-key attention scores (or attention weights). This bias is proportional to the distance between the query and key tokens that comprise each attention score. The distances are scaled by *m*, a head-specific scalar that is fixed during training.

Press et al. found that learning *m* did not lead to strong extrapolation. They instead set *m = (4<sup>(log<sub>2</sub> H + 3)<sup>-1</sup></sup>)<sup>-h</sup>* where *H* is the number of attention heads in a layer and *h* is the index of the current head.

Press et al. report that models trained with ALiBi maintain similar performance even when tested on sequences 5-10x longer than they were trained on. ALiBi’s extrapolation capabilities can be leveraged to train on shorter sequences. This is desirable because the number of operations required to compute self-attention and the GPU memory usage required to store the resulting representations both increase with the square of the sequence length. In one example scenario, Press et al. reported training to equal perplexity in 90% of the time and utilizing 90% of the GPU memory compared to a baseline model with sinusoidal position embeddings. Our experiments showed that ALiBi could reduce perplexity by 0.2-0.6, train models 1.15x faster and utilize 1.2x less GPU memory compared to baseline models (see below).

We conducted experiments on the GPT-2 model family trained on OpenWebText on 8x NVIDIA A100-40GBs. We compared baseline models with learned position embeddings and training sequence length 1024 to models using ALiBi with `train_sequence_length_scaling=0.25` (i.e., train sequence length 256). We found that `train_sequence_length_scaling=0.25` (sequence length 256) provided appreciable speed and accuracy gains for models evaluated at sequence length 1024. Our results are shown in the table below.

|Name|Perplexity|	&Delta;|Train Time (s)|Speedup|GPU Memory|Reduction|
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
|GPT2-52M|30.78||9801||92.91%||
|GPT2-52M ALiBi 0.25x|30.54|-0.24|8411|1.16x|79.79|1.16x|
|GPT2-83M|26.57||17412||97.04||
|GPT2-83M ALiBi 0.25x|26.19|-0.38|14733|1.18x|80.97|1.20x|
|GPT2-125M|24.11||30176||95.96||
|GPT2-125M ALiBi 0.25x|23.49|-0.63|25280|1.19x|74.83|1.28x|

> ❗ Don't Set the Sequence Length Too Short
> 
>We observed that performance significantly degraded for ALiBi models trained on sequence lengths ≤128, implying that very short sequences (≤128 tokens) may be irreconcilably out-of-distribution with regard to longer sequences. Considering our results together with those of Press et al. lead to the suggestion that models with ALiBi should not be trained on sequences ≤256 or `train_sequence_length_scaling≤0.03125`, whichever is larger.

## Attribution

[*Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*](https://openreview.net/forum?id=R8sQPpGCv0) by Ofir Press, Noah A. Smith, and Mike Lewis. Published in ICLR 2022.

*The Composer implementation of this method and the accompanying documentation were produced by Matthew Leavitt at MosaicML.*
