# 🫀 Squeeze-and-Excitation

Tags: `ConvNets`, `Decreased GPU Throughput`, `Increased Accuracy`, `Method`, `Capacity`

## TL;DR

Adds a channel-wise attention operator in CNNs. Attention coefficients are produced by a small, trainable MLP that uses the channels' globally pooled activations as input.

![Squeeze-and-Excitation](https://storage.googleapis.com/docs.mosaicml.com/images/methods/squeeze-and-excitation.png)

After an activation tensor $\mathbf{X}$ is passed through Conv2d $\mathbf{F}_{tr}$ to yield a new tensor $\mathbf{U}$, a Squeeze-Excitation (SE) module scales the channels in a data-dependent manner. The scales are produced by a single-hidden-layer fully-connected network whose input is the global-averaged-pooled $\mathbf{U}$. This can be seen as a channel-wise attention mechanism.

## Attribution

[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) by Jie Hu, Li Shen, and Gang Sun (2018).

## Code and Hyperparameters

- `latent_channels` - Number of channels to use in the hidden layer of MLP that computes channel attention coefficients.
- `min_channels` - The minimum number of output channels in a Conv2d for an SE module to be added afterward.

## Applicable Settings

Applicable to convolutional neural networks. Currently only implemented for CNNs with 2d inputs (e.g., images).

## Example Effects

0.5-1.5%  accuracy gain, roughly 25% slowdown of the model. E.g., we've seen an accuracy change from 76.1 to 77.2% on ImageNet with ResNet-50, in exchange for a training throughput decrease from 4500 samples/sec to 3500 samples/sec on eight RTX 3080 GPUs.

## Implementation Details

Squeeze-Excitation blocks apply channel-wise attention to an activation tensor $\mathbf{X}$. The attention coefficients are produced by a single-hidden-layer MLP (i.e., fully-connected network). This network takes in the result of global average pooling $\mathbf{X}$ as its input vector. In short, the average activations within each channel are used to produce scalar multipliers for each channel.

In order to be architecture-agnostic, our implementation applies the SE attention mechanism after individual conv2d modules, rather than at particular points in particular networks. This results in more SE modules being present than in the original paper.

Our implementation also allows applying the SE module after only certain conv2d modules, based on their channel count (see hyperparameter discussion).

## Suggested Hyperparameters

- `latent_channels` - 64 yielded the best speed-accuracy tradeoffs in our ResNet experiments. The original paper expressed this as a "reduction ratio" $r$ that makes the MLP latent channel count a fraction of the SE block's input channel count. We also support specifying `latent_channels` as a fraction of the input channel count, although we've found that it tends to yield a worse speed vs accuracy tradeoff.
- `min_channels` - For typical CNNs that have lower channel count at higher resolution, this can be used to control where in the network to start applying SE blocks. Ops with higher channel counts take longer to compute relative to the time taken by the SE block. An appropriate value is architecture-dependent, but we weakly suggest setting this to 128 if the architecture in question has modules with at least this many channels.

## Considerations

This method tends to consistently improve the accuracy of CNNs both in absolute terms and when controlling for training and inference time. This may come at the cost of a roughly 20% increase in inference latency, depending on the architecture and inference hardware.

## Composability

Because SE modules slow down the model, they compose well with methods that make the data loader slower (e.g., RandAugment) or that speed up each training step (e.g., Selective Backprop). In the former case, the slower model allows more time for the data loader to run. In the latter case, the initial slowdown allows techniques that accelerate the forward and backward passes to have a greater effect before they become limited by the data loader's speed.


---

## Code
```{eval-rst}
.. autoclass:: composer.algorithms.squeeze_excite.SqueezeExcite
    :members: match, apply
    :noindex:

.. autoclass:: composer.algorithms.squeeze_excite.SqueezeExcite2d
    :noindex:
.. autoclass:: composer.algorithms.squeeze_excite.SqueezeExciteConv2d
    :noindex:
.. autofunction:: composer.algorithms.squeeze_excite.apply_squeeze_excite
    :noindex:
```
