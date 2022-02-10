
# Factorize

![Factorizing a Conv2 operation](https://storage.googleapis.com/docs.mosaicml.com/images/methods/factorize-no-caption.png)

A 2d convolutional layer with {math}`k \times k` filters, {math}`c` input channels and {math}`d` output channels is factorized into two smaller convolutions. The first uses the original filter size, but only {math}`d'` output channels. The second has {math}`1 \times 1` filters and the original number of output channels, but only {math}`d'` input channels. This changes the complexity per output position from {math}`O(k^2 cd)` to {math}`O(k^2c d') + O(d'd)`. Figure from Zhang et al., cited below.

## TL;DR

Splits a large linear or convolutional layer into two smaller ones that compute a similar function.

## Attribution

Factorizing convolution kernels dates back to at least [Gotsman 1994](https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-8659.1320153). To the best of our knowledge, the first papers to apply factorization to modern neural networks were:

 - Jaderberg, Max, Andrea Vedaldi, and Andrew Zisserman. "[Speeding up convolutional neural networks with low rank expansions](https://arxiv.org/abs/1405.3866)." arXiv preprint arXiv:1405.3866 (2014).
 - Denton, Emily L., et al. "[Exploiting linear structure within convolutional networks for efficient evaluation](http://papers.nips.cc/paper/5544-bayesian-inference-for-structured-spike-and-slab-priors.pdf)." Advances in neural information processing systems. 2014.

Our factorization structure most closely matches that of [Zhang et al.](https://ieeexplore.ieee.org/abstract/document/7332968/).


## Hyperparameters

 - `factorize_convs` - whether to factorize conv2d layers
 - `factorize_linears` - whether to factorize linear layers
 - `min_channels` - if a conv2d layer does not have at least
    this many input and output channels, it will not be factorized
- `min_features` - if a linear layer does not have at least
    this many input and output features, it will not be factorized
- `latent_channels` - number of latent channels to use in factorized
    convolutions
- `latent_features` - size of the latent space for factorized linear modules

## Applicable Settings

Factorization can be applied to any model with linear or convolutional layers, but is most likely to be useful for large models with many channels or large hidden layer sizes. At present, only factorizing linear and conv2d modules is supported (i.e., factorizing conv1d and conv3d modules is not supported).

## Example Effects

Based on ResNet-50 experiments, we have not observed factorization to ever be helpful. Even with conservative settings like `min_channels=256`, `latent_channels-128`, we observe over a 1% accuracy loss and a small (<5%) throughput *decrease*, rather than increase.

## Implementation Details

At present, only factorization before training is supported. This is because of limitations of torch [DDP](https://pytorch.org/docs/stable/notes/ddp.html).

We hope to factorization during training in the future. This might allow more intelligent allocation of factorization to different layers based on how well they can be approximated.

## Considerations

Factorization is more likely to be useful for wider operations, meaning higher channel count for convolutions and higher feature count for linear layers. When the layer is not large enough, the execution time is probably limited by memory bandwidth, not compute. Since factorization increases memory bandwidth usage in order to save compute, it is not helpful in this regime.

## Composability

Factorization does not seem to work well even on its own. Since it reduces
model capacity, it is likely to compose especially poorly with other techniques
that reduce model capacity.


## Code
```{eval-rst}
Algorithm
^^^^^^^^^

.. autoclass:: composer.algorithms.factorize.Factorize
    :noindex:

Standalone
^^^^^^^^^^

.. autoclass:: composer.algorithms.factorize.FactorizedConv2d
    :members:
    :noindex:

.. autoclass:: composer.algorithms.factorize.FactorizedLinear
    :members:
    :noindex:

.. autofunction:: composer.algorithms.factorize.apply_factorization
    :noindex:
.. autoclass:: composer.algorithms.factorize.LowRankSolution
    :noindex:
.. autofunction:: composer.algorithms.factorize.factorize_matrix
    :noindex:
.. autofunction:: composer.algorithms.factorize.factorize_conv2d
    :noindex:
```
