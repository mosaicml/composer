# 🕸️ EfficientNet

Category of Task: `Vision`

Kind of Task: `Image Classification`

## Overview

The EfficientNet model family is a set of convolutional neural networks that can be used as the basis for a variety of vision tasks, although they were initially designed for image classification. The model family was designed to reach the highest accuracy for a given computation budget during inference by simultaneously scaling model depth, model width, and image resolution according to an empirically determined scaling law.

## Attribution

Paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) by Mingxing Tan and Quoc V. Le

Code: [gen-efficientnet-pytorch Github repository](https://github.com/rwightman/gen-efficientnet-pytorch) by Ross Wightman

Hyperparameters: [DeepLearningExamples Github repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet) by Nvidia

## Architecture

The table below from Tan and Le specifies the EfficientNet baseline architecture broken up into separate stages. MBConv indicates a mobile inverted bottleneck with a specific expansion size and kernel size. Resolution is the expected input resolution of the current stage. Number of channels is the number of output channels of the current stage. Number of layers indicates the number of repeated blocks in each stage. Subsequent EfficientNet family members scale the resolution, number of channels, and number of layers according to the resolution, width, and depth scaling parameters defined by Tan and Le.

![efficientnet_arch.png](https://storage.googleapis.com/docs.mosaicml.com/images/models/efficientnet_arch.png)

## Family members

Tan and Le included 8 members in their model family. The goal was for each family member to have approximately double the FLOPs of the previous family member. Currently, we only support EfficientNet-B0.

| Model Family Member | Parameter Count | TPU Repo Accuracy* | Our Accuracy** | Training Time on 8x3080 |
|---------------------|-----------------|--------------------|----------------|-------------------------|
| EfficientNet-B0     | 5.3M            | 77.1%              | 77.22%         | 23.3 hr                 |
| EfficientNet-B1     | 7.8M            | 79.1%              | TBA            | TBA                     |
| EfficientNet-B2     | 9.2M            | 80.1%              | TBA            | TBA                     |
| EfficientNet-B3     | 12M             | 81.6%              | TBA            | TBA                     |
| EfficientNet-B4     | 19M             | 82.9%              | TBA            | TBA                     |
| EfficientNet-B5     | 30M             | 83.6%              | TBA            | TBA                     |
| EfficientNet-B6     | 43M             | 84.0%              | TBA            | TBA                     |
| EfficientNet-B7     | 66M             | 84.3%              | TBA            | TBA                     |

*Includes label smoothing, sample-wise stochastic depth, and AutoAugment

**Includes label smoothing and sample-wise stochastic depth

## Default Training Hyperparameters

Our default hyperparameters are identical to the Nvidia Deep Learning Examples except:

- Applying weight decay to batch normalization trainable parameters
- Batch normalization parameters are `momentum = 0.1` and `eps = 1e-5`
