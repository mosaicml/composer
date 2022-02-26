# 📺 Channels Last

![](https://storage.googleapis.com/docs.mosaicml.com/images/methods/channels_last.png)

From [https://developer.nvidia.com/blog/tensor-core-ai-performance-milestones/](https://developer.nvidia.com/blog/tensor-core-ai-performance-milestones/)

AKA: NHWC

Tags: `ConvNets`, `Vision`, `Speedup`, `Best Practice`, `Increased GPU Throughput`

## TL;DR

Channels Last is a systems optimization that improves the throughput of convolution operations by storing activation and weight tensors in a NHWC (batch, height, width, channels) format, rather than Pytorch's default of NCHW.

## Attribution

*[(Beta) Channels Last Memory Format in PyTorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#:~:text=Pytorch%20supports%20memory%20formats%20(and,1%2C%2048%2C%203))* by Vitaly Fedyunin written as part of the PyTorch documentation.

## Applicable Settings

This format applies to two-dimensional convolutional operations, which are typically present in convolutional networks for computer vision like ResNet.

## Hyperparameters

None

## Example Effects

Networks that use this method should be mathematically equivalent to networks that do not use this method (excluding low-level numerical differences or nondeterminism). This method accelerates convolution operations by 1.5x-2x depending on the exact configuration of the layer. Overall, it accelerates ResNet-50 on ImageNet by about 30% on NVIDIA V100 and A100 accelerators.

## Implementation Details

At a high level, NVIDIA tensor cores require tensors to be in NHWC format in order to get the best performance, but PyTorch creates tensors in NCHW format. Every time a convolution operation is called by a layer like `torch.nn.Conv2D`, the cuDNN library performs a transpose to convert the tensor into NHWC format. This transpose introduces overhead.

If the model weights are instead initialized in NHWC format, Pytorch will automatically convert the first input activation tensor to NHWC to match, and it will persist the memory format across all subsequent activations and gradients. This means that convolution operations no longer need to perform transposes, speeding up training.

We currently implement this method by casting the user's model to channels-last format (no changes to the dataloader are necessary). When the first convolution operation receives its input activation, it will automatically convert it to NHWC format, after which the memory format will persist for the remainder of the network (or until it reaches a layer that cannot support having channels last).

## Considerations

If a model has layers that cannot support `channels_last`, there will be overhead due to Pytorch switching activation tensors back and forth between NCHW and NHWC memory formats. We believe this problem currently affects placing channels last on UNet.

## Composability

This method should compose with any other method.

---

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.channels_last.ChannelsLast
    :members: match, apply
    :noindex:

.. autofunction:: composer.algorithms.channels_last.apply_channels_last
    :noindex:
```
