# UNet
[\[Example\]](#example) &middot; [\[Architecture\]](#architecture) &middot; [\[Default Training Hyperparameters\]](#default-training-hyperparameters) &middot; [\[Attribution\]](#attribution) &middot; [\[API Reference\]](#api-reference)

`Vision` / `Segmentation`

Unet is an architecture used for image segmentation.

## Example

<!--pytest-codeblocks:importorskip(monai)-->
<!--pytest-codeblocks:importorskip(scikit-learn)-->
```python
from composer.models import UNet

model = UNet()
```

## Architecture

The figure below ([source](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet)) shows a 3D version of the UNet architecture. Quoting the [Nvidia Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet), "U-Net is composed of a contractive and an expanding path, that aims at building a bottleneck in its centremost part through a combination of convolution, instance norm and leaky relu operations. After this bottleneck, the image is reconstructed through a combination of convolutions and upsampling. Skip connections are added with the goal of helping the backward flow of gradients in order to improve training."

![unet3d.png](https://storage.googleapis.com/docs.mosaicml.com/images/models/unet3d.png)


There are 3 main differences between our implementation and the original NVDA DALI implementation.

The first two refer to removing the NVDA DALI pipeline and replacing all transforms with torch implementations. We are omitting the Zoom transform and use a kernel size of 3 for the Gaussian Blur transform.

While NVDA DLE examples reports the training accuracy using an average of 5 folds, we are using only 1 fold in the interest of faster iteration time, so all of our results are reported using fold 0 and 200 epochs.


## Default Training Hyperparameters

Below are the hyperparameters we used to train UNet on the [BraTS](http://braintumorsegmentation.org) image segmentation dataset.

```yaml
optimizer:
  radam:
    lr: 0.001
    betas: [0.9, 0.999]
    eps: 0.00000001
    weight_decay: 0.0001
schedulers:
  - constant: {}
train_batch_size: 64
max_duration: 200ep
```


## Attribution

The UNet model has been introduced in "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, Thomas Brox in [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597).

We are using the NVDA DLE examples version in
[https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet).

## API Reference

```{eval-rst}
.. autoclass:: composer.models.unet.UNet
    :noindex:
```
