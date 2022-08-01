# 🏙️ ResNet
[\[How to Use\]](#how-to-use) &middot; [\[Architecture\]](#architecture) &middot; [\[Family Members\]](#family-members) &middot; [\[Default Training Hyperparameters\]](#default-training-hyperparameters) &middot; [\[Attribution\]](#attribution) &middot; [\[API Reference\]](#api-reference)

`Vision` / `Image Classification`

The ResNet model family is a set of convolutional neural networks that can be used as a basis for a variety of vision tasks. Our implementation is a simple wrapper on top of the [torchvision ResNet implementation](https://pytorch.org/vision/stable/models.html).

## How to Use

```python
from composer.models import composer_resnet

model = composer_resnet(
    model_name="resnet50",
    num_classes=1000,
    weights=None
)
```

## Architecture

The basic architecture defined in the original papers is as follows:

- The first layer is a 7x7 Convolution with stride 2 and 64 filters.
- Subsequent layers follow 4 stages with {64, 128, 256, 512} input channels with a varying number of residual blocks at each stage that depends on the family member. At the end of every stage, the resolution is reduced by half using a convolution with stride 2.
- The final section consists of a global average pooling followed by a linear + softmax layer that outputs values for the specified number of classes.

The below table from [He et al.](https://arxiv.org/abs/1512.03385) details some of the building blocks for ResNets of different sizes.

![resnet.png](https://storage.googleapis.com/docs.mosaicml.com/images/models/resnet.png)

## Family Members

ResNet family members are identified by their number of layers. Parameter count, accuracy, and training time are provided below.

| Model Family Members | Parameter Count | Our Accuracy | Training Time on 8xA100s |
|----------------------|-----------------|--------------|--------------------------|
| ResNet-18            | 11.5M           | TBA          | TBA                      |
| ResNet-34            | 21.8M           | TBA          | TBA                      |
| ResNet-50            | 25.6M           | 76.5%        | 3.83 hrs                 |
| ResNet-101           | 44.5M           | 78.1%        | 5.50 hrs                 |
| ResNet-152           | 60.2M           | TBA          | TBA                      |


> ❗ **Note**: Please see the [CIFAR ResNet model card](https://docs.mosaicml.com/en/stable/model_cards/cifar_resnet.html#architecture) for the differences between CIFAR and ImageNet ResNets.

## Default Training Hyperparameters

```yaml
optimizer:
  sgd:
    learning_rate: 2.048
    momentum: 0.875
    weight_decay: 5e-4
lr_schedulers:
  linear_warmup: "8ep"
  cosine_decay:
      T_max: "82ep"
      eta_min: 0
      verbose: false
      interval: step
train_batch_size: 2048
max_duration: 90ep
```

## Attribution

Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

Code and hyperparameters: [DeepLearningExamples Github repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) by Nvidia

## API Reference

```{eval-rst}
.. autofunction:: composer.models.resnet.model.composer_resnet
    :noindex:
```
