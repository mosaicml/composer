# ➗ Factorize


[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Computer Vision`  `Natural Language Processing`

Factorize splits a large linear or convolutional layer into two smaller ones that compute a similar function via factorization.
This can be applied to models for both computer vision and natural language processing.

| ![Factorize](https://storage.googleapis.com/docs.mosaicml.com/images/methods/factorize-no-caption.png) |
|:--:
|*The weights `W` of a 2D convolutional layer with `k x k` filters, `c` input chanels, and `d` output channels (a) is factorized into two smaller convolutions (b) with weights `W'` and `p` with `d'` intermediate channels. The first convolution uses the original filter size but produces only `d'` channels. The second convolution has `1 x 1` filters and produces the original `d` output channels but has only `d'` input channels. This changes the complexity from O(k<sup>2</sup>cd) to O(k<sup>2</sup>cd') + O(d'd). This image is a Figure in [https://ieeexplore.ieee.org/abstract/document/7332968](Zhang et al. (2015)).*|

## How to Use

### Functional Interface

```python
# Run the Factorization algorithm directly on the model using the Composer functional API 

from composer import functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())

    # only need to pass in opt if apply_factorization is used after optimizer creation
    # otherwise only the model needs to be passed in
    cf.apply_factorization(model,
                           factorize_convs=True,
                           factorize_linears=True,
                           min_channels=512,
                           latent_channels=0.25,
                           min_features=512,
                           latent_features=0.25,
                           optimizers=opt)

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

from composer.algorithms import Factorize
from composer.trainer import Trainer

factorize = Factorize(factorize_convs=True,
                      factorize_linears=True,
                      min_channels=256,
                      latent_channels=0.25,
                      min_features=256,
                      latent_features=128)

trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  max_duration='1ep',
                  algorithms=[factorize])

trainer.fit()
```

<!--### Implementation Details-->

## Suggested Hyperparameters

We found that `train_sequence_length_scaling=0.25` (sequence length 256) provided appreciable speed and accuracy gains for models evaluated at sequence length 1024.
We observed that performance significantly degraded for ALiBi models trained on sequence lengths ≤128.
As such, we do not recommend training models with sequence lengths ≤256 or `train_sequence_length_scaling≤0.03125`, whichever is larger.

## Technical Details

Based on ResNet-50 experiments, we have not observed Factorize to ever be helpful.
Even with conservative settings like `min_channels=256`, `latent_channels=128`, we observe over a 1% accuracy loss and a small (<5%) throughput decrease, rather than increase.
We have provided this implementation and method card for informational purposes, since factorization is a popular technique in the research literature.

> ❗ There is No Evidence that Factorization Improves Accuracy or Efficiency
> 
> We have only found Factorize to hurt accuracy and reduce throughput.

Factorize is more likely to be useful for wider operations, meaning higher channel count for convolutions and higher feature count for linear layers.
When the layer is not large enough, the execution time is probably limited by memory bandwidth, not compute.
Since factorization increases memory bandwidth usage in order to save compute, it is not helpful in this regime.

At present, only factorization before training is supported. This is because of limitations of PyTorch Distributed Data Parallel.
We hope to allow factorization during training in the future.
This might allow more intelligent allocation of factorization to different layers based on how well they can be approximated.

Factorize can be applied to any model with linear or convolutional layers, but is most likely to be useful for large models with many channels or large hidden layer sizes.
At present, only factorizing `linear` and `conv2d` modules is supported (i.e., factorizing `conv1d` and `conv3d` modules is not supported).

> ❗ Only Linear and 2D Convolutions are Supported
> 
> Factorization does not currently support other kinds of layers, for example 1D and 3D convolutions.

## Attribution

Factorizing convolution kernels dates back to at least Gotsman 1994. To the best of our knowledge, the first papers to apply factorization to modern neural networks were:
* [*Speeding up convolutional neural networks with low rank expansions*](https://arxiv.org/abs/1405.3866) by Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman. Published in the British Machine Vision Conference in 2014.
* [*Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation*](https://arxiv.org/abs/1404.0736) by Emily Denton, Wojciech Zaremba, Joan Bruna, Yann LeCun, and Rob Ferbus. Published in NeurIPS 2014.

Our factorization structure most closely matches that in [*Accelerating Very Deep Convolutional Networks for Classification and Detection*](https://ieeexplore.ieee.org/abstract/document/7332968) by Xiangyu Zhang, Jianhua Zou, Kaiming He, and Jian Sun, which was published in the IEEE Transactions on Pattern Analysis and Machine Intelligence in 2016.

*The Composer implementation of this method and the accompanying documentation were produced by Davis Blalock at MosaicML.*
