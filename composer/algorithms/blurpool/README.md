# 🏊 BlurPool

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

BlurPool increases the accuracy of convolutional neural networks for computer vision at nearly the same speed by applying a spatial low-pass filter before pooling operations and strided convolutions.
Doing so reduces [aliasing](https://en.wikipedia.org/wiki/Aliasing) when performing these operations.

| ![BlurPool](https://storage.googleapis.com/docs.mosaicml.com/images/methods/blurpool-antialiasing.png) |
|:--:
|*A diagram of the BlurPool replacements (bottom row) for typical pooling and downsampling operations (top row) in convolutional neural networks. In each case, BlurPool applies a low-pass filter before the spatial downsampling to avoid aliasing. This image is Figure 2 in [Zhang (2019)](https://proceedings.mlr.press/v97/zhang19a.html).*|

## How to Use

### Functional Interface

```python
# Run the Blurpool algorithm directly on the model using the Composer functional API

import composer.functional as cf
import torch
import torch.nn.functional as F

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())

    # only need to pass in opt if apply_blurpool is used after optimizer
    # creation; otherwise only the model needs to be passed in
    cf.apply_blurpool(model,
                      optimizers=opt,
                      replace_convs=True,
                      replace_maxpools=True,
                      blur_first=True)

    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(10):
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
# The trainer will automatically run it at the appropriate point in the training loop

from composer.algorithms import BlurPool
from composer.trainer import Trainer

def train_model(model, train_dataloader):
    blurpool = BlurPool(replace_convs=True,
                        replace_maxpools=True)
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      max_duration='10ep',
                      algorithms=[blurpool])
    trainer.fit()
```

### Implementation Details

The Composer implementation of BlurPool uses model surgery to replace instances of pooling and downsampling operations with the BlurPool equivalents.
For max pooling, it replaces `torch.nn.MaxPool2d` instances with instances of a custom `nn.Module` subclass that decouples the computation of the max within a given spatial window from the pooling and adds a spatial low-pass filter in between. This change roughly doubles the data movement required for the op, although shouldn’t add significant overhead unless there are many maxpools in the network. For convolutions, it replaces strided `torch.nn.Conv2d` instances (i.e., those where the stride is larger than 1) with a custom module class that 1) applies a low-pass filter to the input, and then 2) applies a copy of the original convolution operation.

## Suggested Hyperparameters

We suggest setting `blur_first=True` to avoid unnecessarily increasing computational cost.
We also suggest setting `blur_maxpools=True` to match the configuration in the original paper, but we haven’t observed a major effect of setting this to either `True` or `False`. We recommend always setting `blur_convs=True` since blurring strided convolutions seems to matter more than blurring maxpools. Note, however, that some models (such as ResNet-20 and other CIFAR resnets) have no strided convolutions, so this argument may have no effect.

## Technical Details

The possible effects of BlurPool can be understood in several different ways:
1. It improves the network’s invariance to small spatial shifts
2. It reduces aliasing in the downsampling operations
3. It adds a structural bias towards preserving low-spatial-frequency components of neural network activations

Consequently, it is likely to be useful on natural images or other inputs that change slowly over their spatial/time dimension(s).

Zhang (2019) showed that BlurPool improves accuracy by 0.5-1% on ImageNet for various networks.
[A follow-up paper by Zou et al.](https://maureenzou.github.io/ddac/) demonstrated similar improvements for ImageNet as well as significant improvements on instance segmentation on MS COCO and semantic segmentation metrics on PASCAL VOC2012 and Cityscapes.
[Lee et al.](https://arxiv.org/abs/2001.06268) also reproduced ImageNet accuracy improvements, especially when applying BlurPool only to strided convolutions.

Depending the value of the `blur_first` parameter, the strided low-pass filtering can happen either before or after the convolution.
Setting `blur_first=True` (i.e., performing low-pass filtering before the convolution) keeps the number of multiply-add operations in the convolution itself constant, adding only the overhead of the low-pass filtering.
Setting `blur_first=False` (i.e., performing low-pass filtering after the convolution) increases the number of multiply-add operations by a factor of `np.prod(conv.stride)` (e.g., 4 for a stride of `(2, 2)`). This more closely matches the approach used in the paper. Anecdotally, we’ve observed this version yielding a roughly 0.1% larger accuracy gain on ResNet-50 + ImageNet in exchange for a ~10% slowdown. Having `blur_first=False` is not as well characterized in our experiments as `blur_first=True`.

> 🚧 Quality/Speed Tradeoff
>
> BlurPool leads to accurracy improvements but also slightly increases training time due to the additional operations it performs.
> On ResNet-50 on ImageNet, we found this tradeoff to be worthwhile: it is a pareto improvement over the standard versions of those benchmarks.
> We also found it to be worthwhile in composition with other methods.
> We recommend that you carefully evaluate whether BlurPool is also a pareto improvement in the context of your application.

Our implementation deviates from the original paper in that we apply the low-pass filter and pooling before the nonlinearity, instead of after. This is because we have no reliable way of adding it after the nonlinearity in an architecture-agnostic way.

BlurPool tends to compose well with other methods. We are not aware of an example of its effects changing significantly as a result of other methods being present.

## Attribution

[*Making Convolutional Networks Shift-Invariant Again*](https://proceedings.mlr.press/v97/zhang19a.html) by Richard Zhang in ICML 2019.

*The Composer implementation of this method and the accompanying documentation were produced by Davis Blalock at MosaicML. We thank Richard Zhang for helpful discussion*
