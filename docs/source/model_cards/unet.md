# ↩️ UNet

Category of Task: `Vision`

Kind of Task: `Segmentation`

Link to Code: [https://github.com/mosaicml/composer/tree/main/composer/models/unet](https://github.com/mosaicml/composer/tree/main/composer/models/unet)

## Overview

UNet is an architecture used in image segmentation. The example we are using is for medical brain tumor data.

## Attribution

The UNet model has been introduced in "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, Thomas Brox in [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597).

We are using the NVDA DLE examples version in
[https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet).

## Architecture

The figure below shows a 3D version of the UNet architecture. Quoting the DLE examples, U-Net is composed of a contractive and an expanding path that aims at building a bottleneck in its centermost part through a combination of convolution, instance norm, and leaky relu operations. After this bottleneck, the image is reconstructed through a combination of convolutions and upsampling. Skip connections are added with the goal of helping the backward flow of gradients in order to improve training.

![unet3d.png](https://storage.googleapis.com/docs.mosaicml.com/images/models/unet3d.png)

## Implementation Details

There are 3 main differences between our implementation and the original NVDA DALI implementation.

The first two refer to removing the NVDA DALI pipeline and replacing all transforms with torch implementations. We are omitting the Zoom transform and use a kernel size of 3 for the Gaussian Blur transform.

While NVDA DLE examples reports the training accuracy using an average of 5 folds, we are using only 1 fold in the interest of faster iteration time, so all of our results are reported using fold 0 and 200 epochs.

## Exploring Tradeoffs Between Quality and Training Speed/Cost

As noted above, we are reporting only 1 fold and a fixed number of 200 epochs in training the model, while DLE uses early stopping.
