# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""DeepLabV3 model extending :class:`.ComposerClassifier`."""

import functools
import textwrap
import warnings
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
import torchvision
from packaging import version
from torchmetrics import MetricCollection
from torchvision.models import _utils, resnet

from composer.loss import DiceLoss, soft_cross_entropy
from composer.metrics import CrossEntropy, MIoU
from composer.models.initializers import Initializer
from composer.models.tasks import ComposerClassifier
from composer.utils import dist

__all__ = ['deeplabv3', 'composer_deeplabv3']


class SimpleSegmentationModel(torch.nn.Module):

    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        logits = self.classifier(tuple(features.values()))
        logits = F.interpolate(logits,
                               size=input_shape,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=False)
        return logits


def deeplabv3(num_classes: int,
              backbone_arch: str = 'resnet101',
              backbone_weights: Optional[str] = None,
              sync_bn: bool = True,
              use_plus: bool = True,
              initializers: Sequence[Initializer] = ()):
    """Helper function to build a mmsegmentation DeepLabV3 model.

    Args:
        num_classes (int): Number of classes in the segmentation task.
        backbone_arch (str, optional): The architecture to use for the backbone. Must be either
            [``'resnet50'``, ``'resnet101'``]. Default: ``'resnet101'``.
        backbone_weights (str, optional): If specified, the PyTorch pre-trained weights to load for the backbone.
            Currently, only ['IMAGENET1K_V1', 'IMAGENET1K_V2'] are supported. Default: ``None``.
        sync_bn (bool, optional): If ``True``, replace all BatchNorm layers with SyncBatchNorm layers.
            Default: ``True``.
        use_plus (bool, optional): If ``True``, use DeepLabv3+ head instead of DeepLabv3. Default: ``True``.
        initializers (Sequence[Initializer], optional): Initializers for the model. ``()`` for no initialization.
            Default: ``()``.

    Returns:
        deeplabv3: A DeepLabV3 :class:`torch.nn.Module`.

    Example:

    .. code-block:: python

        from composer.models.deeplabv3.deeplabv3 import deeplabv3

        pytorch_model = deeplabv3(num_classes=150, backbone_arch='resnet101', backbone_weights=None)
    """

    # check that the specified architecture is in the resnet module
    if not hasattr(resnet, backbone_arch):
        raise ValueError(f'backbone_arch must be part of the torchvision resnet module, got value: {backbone_arch}')

    # change the model weight url if specified
    if version.parse(torchvision.__version__) < version.parse('0.13.0'):
        pretrained = False
        if backbone_weights:
            pretrained = True
            if backbone_weights == 'IMAGENET1K_V1':
                resnet.model_urls[backbone_arch] = 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
            elif backbone_weights == 'IMAGENET1K_V2':
                resnet.model_urls[backbone_arch] = 'https://download.pytorch.org/models/resnet101-cd907fc2.pth'
            else:
                ValueError(
                    textwrap.dedent(f"""\
                        `backbone_weights` must be either "IMAGENET1K_V1" or "IMAGENET1K_V2"
                        if torchvision.__version__ < 0.13.0. `backbone_weights` was {backbone_weights}."""))
        backbone = getattr(resnet, backbone_arch)(pretrained=pretrained,
                                                  replace_stride_with_dilation=[False, True, True])
    else:
        backbone = getattr(resnet, backbone_arch)(weights=backbone_weights,
                                                  replace_stride_with_dilation=[False, True, True])

    # specify which layers to extract activations from
    return_layers = {'layer1': 'layer1', 'layer4': 'layer4'} if use_plus else {'layer4': 'layer4'}
    backbone = _utils.IntermediateLayerGetter(backbone, return_layers=return_layers)

    try:
        from mmseg.models import ASPPHead, DepthwiseSeparableASPPHead
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Either mmcv or mmsegmentation is not installed. To install mmcv, please run pip install mmcv-full==1.4.4 -f
             https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html where {cu_version} and
             {torch_version} refer to your CUDA and PyTorch versions, respectively. To install mmsegmentation, please
             run pip install mmsegmentation==0.22.0 on command-line.""")) from e

    world_size = dist.get_world_size()
    if sync_bn and world_size == 1:
        warnings.warn('sync_bn was true, but only one process is present for training. sync_bn will be ignored.')

    norm_type = 'SyncBN' if sync_bn and world_size > 1 else 'BN'
    norm_cfg = {'type': norm_type, 'requires_grad': True}
    if use_plus:
        # mmseg config:
        # https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/models/deeplabv3plus_r50-d8.py
        head = DepthwiseSeparableASPPHead(in_channels=2048,
                                          in_index=-1,
                                          channels=512,
                                          dilations=(1, 12, 24, 36),
                                          c1_in_channels=256,
                                          c1_channels=48,
                                          dropout_ratio=0.1,
                                          num_classes=num_classes,
                                          norm_cfg=norm_cfg,
                                          align_corners=False)
    else:
        # mmseg config:
        # https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/models/deeplabv3_r50-d8.py
        head = ASPPHead(in_channels=2048,
                        in_index=-1,
                        channels=512,
                        dilations=(1, 12, 24, 36),
                        dropout_ratio=0.1,
                        num_classes=num_classes,
                        norm_cfg=norm_cfg,
                        align_corners=False)

    model = SimpleSegmentationModel(backbone, head)

    if initializers:
        for initializer in initializers:
            initializer_fn = Initializer(initializer).get_initializer()

            # Only apply initialization to classifier head if pre-trained weights are used
            if backbone_weights is None:
                model.apply(initializer_fn)
            else:
                model.classifier.apply(initializer_fn)

    if sync_bn and world_size > 1:
        local_world_size = dist.get_local_world_size()

        # List of ranks for each node, assumes that each node has the same number of ranks
        num_nodes = world_size // local_world_size
        process_group = None
        if num_nodes > 1:
            ranks_per_node = [
                list(range(node * local_world_size, (node + 1) * local_world_size)) for node in range(num_nodes)
            ]
            process_groups = [torch_dist.new_group(ranks) for ranks in ranks_per_node]
            process_group = process_groups[dist.get_node_rank()]

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=process_group)

    return model


def composer_deeplabv3(num_classes: int,
                       backbone_arch: str = 'resnet101',
                       backbone_weights: Optional[str] = None,
                       sync_bn: bool = True,
                       use_plus: bool = True,
                       ignore_index: int = -1,
                       cross_entropy_weight: float = 1.0,
                       dice_weight: float = 0.0,
                       initializers: Sequence[Initializer] = ()):
    """Helper function to create a :class:`.ComposerClassifier` with a DeepLabv3(+) model. Logs
        Mean Intersection over Union (MIoU) and Cross Entropy during training and validation.

    From `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`_
        (Chen et al, 2017).

    Args:
        num_classes (int): Number of classes in the segmentation task.
        backbone_arch (str, optional): The architecture to use for the backbone. Must be either
            [``'resnet50'``, ``'resnet101'``]. Default: ``'resnet101'``.
        backbone_weights (str, optional): If specified, the PyTorch pre-trained weights to load for the backbone.
            Currently, only ['IMAGENET1K_V1', 'IMAGENET1K_V2'] are supported. Default: ``None``.
        sync_bn (bool, optional): If ``True``, replace all BatchNorm layers with SyncBatchNorm layers.
            Default: ``True``.
        use_plus (bool, optional): If ``True``, use DeepLabv3+ head instead of DeepLabv3. Default: ``True``.
        ignore_index (int): Class label to ignore when calculating the loss and other metrics. Default: ``-1``.
        cross_entropy_weight (float): Weight to scale the cross entropy loss. Default: ``1.0``.
        dice_weight (float): Weight to scale the dice loss. Default: ``0.0``.
        initializers (List[Initializer], optional): Initializers for the model. ``[]`` for no initialization.
            Default: ``[]``.


    Returns:
        ComposerModel: instance of :class:`.ComposerClassifier` with a DeepLabv3(+) model.

    Example:

    .. code-block:: python

        from composer.models import composer_deeplabv3

        model = composer_deeplabv3(num_classes=150, backbone_arch='resnet101', backbone_weights=None)
    """

    model = deeplabv3(backbone_arch=backbone_arch,
                      backbone_weights=backbone_weights,
                      use_plus=use_plus,
                      num_classes=num_classes,
                      sync_bn=sync_bn,
                      initializers=initializers)

    train_metrics = MetricCollection(
        [CrossEntropy(ignore_index=ignore_index),
         MIoU(num_classes, ignore_index=ignore_index)])
    val_metrics = MetricCollection(
        [CrossEntropy(ignore_index=ignore_index),
         MIoU(num_classes, ignore_index=ignore_index)])

    ce_loss_fn = functools.partial(soft_cross_entropy, ignore_index=ignore_index)
    dice_loss_fn = DiceLoss(softmax=True, batch=True, ignore_absent_classes=True)

    def _combo_loss(output, target) -> Dict[str, torch.Tensor]:
        loss = {'total': torch.zeros(1, device=output.device, dtype=output.dtype)}
        if cross_entropy_weight:
            loss['cross_entropy'] = ce_loss_fn(output, target)
            loss['total'] += loss['cross_entropy'] * cross_entropy_weight
        if dice_weight:
            loss['dice'] = dice_loss_fn(output, target)
            loss['total'] += loss['dice'] * dice_weight
        return loss

    composer_model = ComposerClassifier(module=model,
                                        train_metrics=train_metrics,
                                        val_metrics=val_metrics,
                                        loss_fn=_combo_loss)
    return composer_model
