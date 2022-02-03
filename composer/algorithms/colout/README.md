# ColOut `Vision`

ColOut is a data augmentation technique that drops a fraction of the rows or columns of an input image for a computer vision model.
If the fraction of rows/columns isn't too large, the image content is not significantly altered but the image size is reduced, speeding up training.
This modification modestly reduces accuracy, but it is a worthwhile tradeoff for the improved speed.

| ![ColOut](col_out.png) |
|:--:
|*Several instances of an image of an apple from the CIFAR-100 dataset with ColOut applied.*|

## How to Use

### Functional Interface

```python
def training_loop(model, train_loader):
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

## Technical Details

ALiBi dispenses with traditional position embeddings and instead adds a static, non-learned bias to the query-key attention scores (or attention weights). This bias is proportional to the distance between the query and key tokens that comprise each attention score. The distances are scaled by *m*, a head-specific scalar that is fixed during training.

Press et al. found that learning *m* did not lead to strong extrapolation. They instead set *m = (4<sup>(log<sub>2</sub> H + 3)<sup>-1</sup></sup>)<sup>-h</sup>* where *H* is the number of attention heads in a layer and *h* is the index of the current head.

Press et al. report that models trained with ALiBi maintain similar performance even when tested on sequences 5-10x longer than they were trained on. ALiBi’s extrapolation capabilities can be leveraged to train on shorter sequences. This is desirable because the number of operations required to compute self-attention and the GPU memory usage required to store the resulting representations both increase with the square of the sequence length. In one example scenario, Press et al. reported training to equal perplexity in 90% of the time and utilizing 90% of the GPU memory compared to a baseline model with sinusoidal position embeddings. Our experiments showed that ALiBi could reduce perplexity by 0.2-0.6, train models 1.15x faster and utilize 1.2x less GPU memory compared to baseline models (see below).

> 👍 Suggested Hyperparameters
> 
> We found that `train_sequence_length_scaling=0.25` (sequence length 256) provided appreciable speed and accuracy gains for models evaluated at sequence length 1024.

We conducted experiments on the GPT-2 model family trained on OpenWebText on 8x NVIDIA A100-40GBs. We compared baseline models with learned position embeddings and training sequence length 1024 to models using ALiBi with `train_sequence_length_scaling=0.25` (i.e., train sequence length 256). Our results are shown in the table below.

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
