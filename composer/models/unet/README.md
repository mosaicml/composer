## Overview

Unet is an example of architecture used in image segmentation. The example we are using is for medical brain tumor data, specifically the BRATS dataset.

## Attribution

The UNet model has been introduced in "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, Thomas Brox in [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597). 

We are using the 2D version model in the NVDA DLE examples in 
[https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet).

For guidance on getting the data and preprocessing it, please follow
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet#getting-the-data.

## Implementation Details

There are 3 main differences between our implementation and the original NVDA DALI implementation. 

The first two refer to removing the NVDA DALI pipeline and replacing all transforms with torch implementations. We are omitting the Zoom transform and use a kernel size of 3 for the Gaussian Blur transform.

While NVDA DLE examples reports the training accuracy using an average of 5 folds, we are using only 1 fold in the interest of faster iteration time, so all of our results are reported using fold 0 and 200 epochs.