# 🏊 BlurPool

![BlurPool Antialiasing](https://storage.googleapis.com/docs.mosaicml.com/images/methods/blurpool-antialiasing.png)

How various original ops (top row) are replaced with corresponding BlurPool ops (bottom row) in the original paper. In each case, a [low-pass filter](https://en.wikipedia.org/wiki/Low-pass_filter) is applied before the spatial downsampling to avoid [aliasing](https://en.wikipedia.org/wiki/Aliasing).

Tags: `Vision`, `Decreased GPU Throughput`, `Increased Accuracy`, `Method`, `Regularization`

## TL;DR

Increases accuracy at nearly the same speed by applying a spatial low-pass filter before the pool in max pooling and whenever using a strided convolution.

## Attribution

[Making Convolutional Networks Shift-Invariant Again.](https://proceedings.mlr.press/v97/zhang19a.html) by Richard Zhang (2019).

[Implementation by Richard Zhang (GitHub)](https://github.com/adobe/antialiased-cnns)

[Project Website by Richard Zhang](https://richzhang.github.io/antialiased-cnns/)

## Code and Hyperparameters

- `replace_convs` - replace strided `torch.nn.Conv2d` modules within the module with anti-aliased versions
- `replace_maxpools` - replace torch.nn.MaxPool2d modules with anti-aliased versions
- `blur_first` - when `replace_convs`  is `True`, blur input before the associated convolution. When set to `False`, the convolution is applied with a stride of 1 before the blurring, resulting in significant overhead (though more closely matching the paper).

## Applicable Settings

Applicable whenever using a strided convolution or a local max pooling layer, which mainly occur in vision settings. We have currently implemented it for the PyTorch operators MaxPool2d and Conv2d.

## Example Effects

The [original paper](https://arxiv.org/abs/1904.11486) showed accuracy gains of around 0.5-1% on ImageNet for various networks. A [subsequent paper](https://maureenzou.github.io/ddac/) demonstrated similar gains for ImageNet, as well as significant improvements on instance segmentation on MS COCO. The latter paper also showed improvements in semantic segmentation metrics on PASCAL VOC2012 and Cityscapes. [Lee et al.](https://arxiv.org/abs/2001.06268) have also reproduced ImageNet accuracy gains, especially when applying Blurpool only to strided convolutions.

## Implementation Details

For max pooling, we replace `torch.nn.MaxPool2d` instances with instances of a custom `nn.Module` subclass that decouples the computation of the max within a given spatial window from the pooling and adds a spatial low-pass filter in between. This change roughly doubles the data movement required for the op, although shouldn't add significant overhead unless there are many maxpools in the network.

For convolutions, we replace strided `torch.nn.Conv2d` instances with a custom module class that 1) applies a low-pass filter to the input, and 2) applies a copy of the original convolution operation. Depending the value of the `blur_first` parameter, the strided low-pass filtering can happen either before or after the convolution. The former keeps the number of multiply-add operations in the convolution itself constant, and only adds the overhead of the low-pass filtering. The latter increases the number of multiply-add operations by a factor of `np.prod(conv.stride)`  (e.g., 4 for a stride of `(2, 2)`). This more closely matches the approach used in the paper.  Anecdotally, we've observed this version yielding a roughly 0.1% accuracy gain on ResNet-50 + ImageNet in exchange for a ~10% slowdown. Having `blur_first=False` is not as well characterized in our experiments as `blur_first=True`.

Our implementation deviates from the original paper in that we apply the low-pass filter and pooling before the nonlinearity, instead of after. This is because we (so far) have no reliable way of adding it after the nonlinearity in an architecture-agnostic way.

## Suggested Hyperparameters

We weakly suggest setting `blur_maxpools=True`to match the configuration in the paper, since we haven't observed a large benefit either way.

We suggest setting `blur_first=True` to avoid increased computational cost.

## Considerations

This method can be understood in several ways:

1. It improves the network's invariance to small spatial shifts
2. It reduces aliasing in the downsampling operations
3. It adds a structural bias towards preserving low-spatial-frequency components of neural network activations

Consequently, it is likely to be useful on natural images, or other inputs that change slowly over their spatial/time dimension(s).

## Composability

BlurPool tends to compose well with other methods. We are not aware of an example of its effects changing significantly as a result of other methods being present.

## Acknowledgments

We thank Richard Zhang for helpful discussion.

## Code
```{eval-rst}
.. autoclass:: composer.algorithms.blurpool.BlurPool
    :members: match, apply
    :noindex:
.. autoclass:: composer.algorithms.blurpool.BlurConv2d
    :noindex:
.. autoclass:: composer.algorithms.blurpool.BlurMaxPool2d
    :noindex:
.. autoclass:: composer.algorithms.blurpool.BlurPool2d
    :noindex:
.. autofunction:: composer.algorithms.blurpool.blur_2d
    :noindex:
.. autofunction:: composer.algorithms.blurpool.blurmax_pool2d
    :noindex:
.. autofunction:: composer.algorithms.blurpool.apply_blurpool
    :noindex:
```
