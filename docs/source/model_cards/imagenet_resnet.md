# ImageNet ResNet

Category of Task: `Vision`

Kind of Task: `Image Classification`

## Overview

The ResNet model family is a set of convolutional neural networks that can be used as the base for a variety of vision tasks. ImageNet ResNets are a subset of the ResNet family which were designed specifically for classification on the ImageNet dataset. 

## Attribution

Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

Code and hyperparameters: [DeepLearningExamples Github repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) by Nvidia

## Architecture

The basic architecture defined in the original papers is as follows:

- The first layer is a 7x7 Convolution with stride 2 and 64 filters.
- Subsequent layers follow 4 stages with {64, 128, 256, 512} input channels with a varying number of residual blocks at each stage that depends on the family member. At the end of every stage, the resolution is reduced by half using a convolution with stride 2.
- The final section consists of a global average pooling followed by a linear + softmax layer that outputs values for the specified number of classes

## Family Members

ResNet family members are defined by their number of layers. Parameter count, accuracy, and training time are provided below.

| Model Family Members | Parameter Count | Our Accuracy | Training Time on 8xA100s |
|----------------------|-----------------|--------------|--------------------------|
| ResNet-18            | 11.5M           | TBA          | TBA                      |
| ResNet-34            | 21.8M           | TBA          | TBA                      |
| ResNet-50            | 25.6M           | 76.5%        | 3.83 hrs                 |
| ResNet-101           | 44.5M           | 78.1%        | 5.50 hrs                 |
| ResNet-152           | 60.2M           | TBA          | TBA                      |

## Implementation details

The implementation follows He et al. except for the ResNet v1.5 change from Nvidia's DeepLearningExamples. For bottleneck residual blocks that downsample, the downsampling is performed by the 3x3 convolution instead of the 1x1 convolution. Nvidia reports approximately +0.5% absolute accuracy and 5% decrease in throughput from this change. 

## Default Training Hyperparameters

- Optimizer: SGD
    - Learning rate: 2.048
    - Momentum: 0.875
    - Weight decay: 1/32768
- Batch size: 2048
- LR Schedulers
    - Linear warmup for 8 epochs
    - Cosine decay after warmup
- Number of epochs: 90