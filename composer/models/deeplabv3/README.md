# 🤿 DeepLabv3+
[\[Example\]](#example) &middot; [\[Architecture\]](#architecture) &middot; [\[Training Hyperparameters\]](#training-hyperparameters) &middot; [\[Attribution\]](#attribution) &middot; [\[API Reference\]](#api-reference)

[DeepLabv3+](https://arxiv.org/abs/1802.02611) is an architecture designed for semantic segmenation i.e. per-pixel classification. DeepLabv3+ takes in a feature map from a backbone architecture (e.g. ResNet-101), then outputs classifications for each pixel in the input image. Our implementation is a simple wrapper around [torchvision’s ResNet](https://pytorch.org/vision/stable/models.html#id10) for the backbone and [mmsegmentation’s DeepLabv3+](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3plus) for the head.

## Example

<!--pytest.mark.skip-->
```python
from composer.models import composer_deeplabv3

model = composer_deeplabv3(num_classes=150,
                           backbone_arch="resnet101",
                           is_backbone_pretrained=True,
                           backbone_url="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
                           sync_bn=False
)
```

## Architecture

Based on [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

<div align=center>
<img src="https://storage.googleapis.com/docs.mosaicml.com/images/models/deeplabv3_v2.png" alt="deeplabv3plus" width="650">
</div>


- **Backbone network**: converts the input image into a feature map.
    * Usually ResNet-101 with the strided convolutions converted to dilations convolutions in stage 3 and 4.
    * The 3x3 convolutions in stage 3 and 4 have dilation sizes of 2 and 4, respectively, to compensate for the decreased receptive field.
    * The average pooling and classification layer are ignored.
- **Spatial Pyramid Pooling**: extracts multi-resolution features from the stage 4 backbone feature map.
    * The backbone feature map is processed with four parallel convolution layers with dilations {1, 12, 24, 36} and kernel sizes {1x1, 3x3, 3x3, 3x3}.
    * In parallel to the convolutions, global average pool the backbone feature map, then bilinearly upsample to be the same spatial dimension as the feature map.
    * Concatenate the outputs from the convolutions and global average pool, then process with a 1x1 convolution.
    * The 3x3 convolutions are implemented as depth-wise convolutions to reduce memory and computation cost.
- **Decoder**: converts the output of spatial pyramid pooling (SPP) to class predictions of the same spatial dimension as the input image.
    * SPP output is bilinearly upsampled to be the same spatial dimension as the output from the first stage in the backbone network.
    * A 1x1 convolution is applied to the first stage activations, then this is concatenated with the upsampled SPP output.
    * The concatenation is processed by a 3x3 convolution with dropout followed by a classification layer.
    * The predictions are bilinearly upsampled to be the same resolution as the input image.

## Training Hyperparameters

We tested two sets of hyperparameters for DeepLabv3+ trained on the ADE20k dataset.

### Typical ADE20k Model Hyperparameters

```yaml
model:
  deeplabv3:
    initializers:
      - kaiming_normal
      - bn_ones
    num_classes: 150
    backbone_arch: resnet101
    is_backbone_pretrained: true
    use_plus: true
    sync_bn: true
optimizer:
  sgd:
    lr: 0.01
    momentum: 0.9
    weight_decay: 5.0e-4
    dampening: 0
    nesterov: false
schedulers:
  - polynomial:
      alpha_f: 0.01
      power: 0.9
max_duration: 127ep
train_batch_size: 16
precision: amp
```

| Model | mIoU | Time-to-Train on 8xA100 |
| --- | --- | --- |
| ResNet101-DeepLabv3+ | 44.17 +/- 0.17 | 6.385 hr |

### Composer ADE20k Model Hyperparameters

```yaml
model:
  deeplabv3:
    initializers:
      - kaiming_normal
      - bn_ones
    num_classes: 150
    backbone_arch: resnet101
    is_backbone_pretrained: true
    use_plus: true
    sync_bn: true
    # New Pytorch pretrained weights
    backbone_url: https://download.pytorch.org/models/resnet101-cd907fc2.pth
optimizer:
  decoupled_sgdw:
    lr: 0.01
    momentum: 0.9
    weight_decay: 2.0e-5
    dampening: 0
    nesterov: false
schedulers:
  - cosine_decay:
      t_max: 1dur
max_duration: 128ep
train_batch_size: 32
precision: amp
```

| Model | mIoU | Time-to-Train on 8xA100 |
| --- | --- | --- |
| ResNet101-DeepLabv3+ | 45.764 +/- 0.29 | 4.67 hr |

Improvements:

- New PyTorch pretrained weights
- Cosine decay
- Decoupled Weight Decay
- Increase batch size to 32
- Decrease weight decay to 2e-5

## Attribution

[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) by Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam

[OpenMMLab Semantic Segmentation Toolbox and Benchmark](https://github.com/open-mmlab/mmsegmentation)

[How to Train State-Of-The-Art Models Using TorchVision’s Latest Primitives](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/) by Vasilis Vryniotis

## API Reference

```{eval-rst}
.. autoclass:: composer.models.deeplabv3.composer_deeplabv3
    :noindex:
```
