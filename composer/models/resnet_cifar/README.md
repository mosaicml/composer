# CIFAR ResNet
[\[Example\]](#example) &middot; [\[Architecture\]](#architecture) &middot; [\[Family Members\]](#family-members) &middot; [\[Default Training Hyperparameters\]](#default-training-hyperparameters) &middot; [\[Attribution\]](#attribution) &middot; [\[API Reference\]](#api-reference)

`Vision` / `Image Classification`

The ResNet model family is a set of convolutional neural networks that can be used as the basis for a variety of vision tasks. CIFAR ResNet models are a subset of this family designed specifically for the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets.

## Example

```python
from composer.models import composer_resnet_cifar

model = composer_resnet_cifar(model_name='resnet_56', num_classes=10)
```

## Architecture

Residual Networks are feedforward convolutional networks with “residual” connections between non-consecutive layers.

The model architecture is defined by the original paper:

- The network inputs are of dimension 32×32x3.
- The first layer is 3×3 convolutions
- The subsequent layers are a stack of 6n layers with 3×3 convolutions on the feature maps of sizes {32,16,8}, with 2n layers for each feature map size. The number of filters are {16,32,64} for the respective feature map sizes. Subsampling is performed by convolutions with a stride of 2
- The network ends with a global average pooling, a linear layer with the output dimension equal to the number of classes, and softmax function.

There are a total 6n+2 stacked weighted layers. Each family member is specified by the number of layers, for example n=9 corresponds to ResNet56

The biggest differences between CIFAR ResNet models and ImageNet ResNet models are:

- CIFAR ResNet models use fewer filters for each convolution.
- The ImageNet ResNets contain four stages, while the CIFAR ResNets contain three stages. In addition, CIFAR ResNets uniformly distribute blocks across each stage while ImageNet ResNets have a specific number of blocks for each stage.

## Family Members

| Model Family Members | Parameter Count | Our Accuracy | Training Time on 1x3080 |
|----------------------|-----------------|--------------|-------------------------|
| ResNet20             | 0.27M           | TBA          | TBA                     |
| ResNet32             | 0.46M           | TBA          | TBA                     |
| ResNet44             | 0.66M           | TBA          | TBA                     |
| ResNet56             | 0.85M           | 93.1%        | 35 min                  |
| ResNet110            | 1.7M            | TBA          | TBA                     |
## Default Training Hyperparameters

```yaml
optimizer:
  sgd:
    learning_rate: 1.2
    momentum: 0.9
    weight_decay: 1e-4
schedulers:
  - multistep_with_warmup:
      t_warmup: "5ep"
      milestones:
        - "80ep"
        - "120ep"
      gamma: 0.1
train_batch_size: 1024
max_duration: 160ep
```

## Attribution

Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

Note that this paper set the standard for ResNet style architectures for both CIFAR-10/100 and ImageNet

## API Reference

```{eval-rst}
.. autoclass:: composer.models.resnet_cifar.model.composer_resnet_cifar
    :noindex:
```
