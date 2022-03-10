# Copyright 2021 MosaicML. All Rights Reserved.

"""DeepLabV3 model extending :class:`.ComposerClassifier`."""

import textwrap
from typing import Any, List

import torch
import torch.nn.functional as F
from torchmetrics.collections import MetricCollection
from torchvision.models import _utils, resnet

from composer.core.types import BatchPair
from composer.models.base import ComposerModel
from composer.models.loss import CrossEntropyLoss, MIoU, soft_cross_entropy
from composer.models.model_hparams import Initializer

__all__ = ["deeplabv3_builder", "ComposerDeepLabV3"]


class SimpleSegmentationModel(torch.nn.Module):

    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        logits = self.classifier(tuple(features.values()))
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        return logits


def deeplabv3_builder(num_classes: int,
                      backbone_arch: str = 'resnet101',
                      is_backbone_pretrained: bool = True,
                      backbone_url: str = '',
                      sync_bn: bool = True,
                      use_plus: bool = True,
                      initializers: List[Initializer] = []):
    """Helper function to build a torchvision DeepLabV3 model with a 3x3 convolution layer and dropout removed.

    Args:
        num_classes (int): Number of classes in the segmentation task.
        backbone_arch (str, optional): The architecture to use for the backbone. Must be either [``'resnet50'``, ``'resnet101'``].
            Default: ``'resnet101'``.
        is_backbone_pretrained (bool, optional): If ``True``, use pretrained weights for the backbone. Default: ``True``.
        backbone_url (str, optional): Url used to download model weights. If empty, the PyTorch url will be used.
            Default: ``''``.
        sync_bn (bool, optional): If ``True``, replace all BatchNorm layers with SyncBatchNorm layers. Default: ``True``.
        use_plus (bool, optional): If ``True``, use DeepLabv3+ head instead of DeepLabv3. Default: ``True``.
        initializers (List[Initializer], optional): Initializers for the model. ``[]`` for no initialization. Default: ``[]``.

    Returns:
        deeplabv3: A DeepLabV3 :class:`torch.nn.Module`.

    Example:

    .. code-block:: python

        from composer.models.deeplabv3.deeplabv3 import deeplabv3_builder

        pytorch_model = deeplabv3_builder(num_classes=150, backbone_arch='resnet101', is_backbone_pretrained=False)
    """

    # check that the specified architecture is in the resnet module
    if not hasattr(resnet, backbone_arch):
        raise ValueError(f"backbone_arch must be part of the torchvision resnet module, got value: {backbone_arch}")

    # change the model weight url if specified
    if backbone_url:
        resnet.model_urls[backbone_arch] = backbone_url
    backbone = getattr(resnet, backbone_arch)(pretrained=is_backbone_pretrained,
                                              replace_stride_with_dilation=[False, True, True])

    # specify which layers to extract activations from
    return_layers = {'layer1': 'layer1', 'layer4': 'layer4'} if use_plus else {'layer4': 'layer4'}
    backbone = _utils.IntermediateLayerGetter(backbone, return_layers=return_layers)

    try:
        from mmseg.models import ASPPHead, DepthwiseSeparableASPPHead  # type: ignore
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Either mmcv or mmsegmentation is not installed. To install mmcv, please run pip install mmcv-full==1.4.4 -f
             https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html where {cu_version} and
             {torch_version} refer to your CUDA and PyTorch versions, respectively. To install mmsegmentation, please
             run pip install mmsegmentation==0.22.0 on command-line.""")) from e
    norm_cfg = dict(type='SyncBN', requires_grad=True)
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
            if is_backbone_pretrained:
                model.classifier.apply(initializer_fn)
            else:
                model.apply(initializer_fn)

    if sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model


class ComposerDeepLabV3(ComposerModel):
    """DeepLabV3 model extending :class:`.ComposerClassifier`. Logs Mean Intersection over Union (MIoU) and Cross
    Entropy during training and validation.

    From `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`_ (Chen et al, 2017).

    Args:
        num_classes (int): Number of classes in the segmentation task.
        backbone_arch (str, optional): The architecture to use for the backbone. Must be either [``'resnet50'``, ``'resnet101'``].
            Default: ``'resnet101'``.
        is_backbone_pretrained (bool, optional): If ``True``, use pretrained weights for the backbone. Default: ``True``.
        backbone_url (str, optional): Url used to download model weights. If empty, the PyTorch url will be used.
            Default: ``''``.
        sync_bn (bool, optional): If ``True``, replace all BatchNorm layers with SyncBatchNorm layers. Default: ``True``.
        use_plus (bool, optional): If ``True``, use DeepLabv3+ head instead of DeepLabv3. Default: ``True``.
        initializers (List[Initializer], optional): Initializers for the model. ``[]`` for no initialization. Default: ``[]``.


    Example:

    .. code-block:: python

        from composer.models import ComposerDeepLabV3

        model = ComposerDeepLabV3(num_classes=150, backbone_arch='resnet101', is_backbone_pretrained=False)
    """

    def __init__(self,
                 num_classes: int,
                 backbone_arch: str = 'resnet101',
                 is_backbone_pretrained: bool = True,
                 backbone_url: str = '',
                 sync_bn: bool = True,
                 use_plus: bool = True,
                 initializers: List[Initializer] = []):

        super().__init__()
        self.num_classes = num_classes
        self.model = deeplabv3_builder(backbone_arch=backbone_arch,
                                       is_backbone_pretrained=is_backbone_pretrained,
                                       backbone_url=backbone_url,
                                       use_plus=use_plus,
                                       num_classes=num_classes,
                                       sync_bn=sync_bn,
                                       initializers=initializers)

        # Metrics
        self.train_miou = MIoU(self.num_classes, ignore_index=-1)
        self.train_ce = CrossEntropyLoss(ignore_index=-1)
        self.val_miou = MIoU(self.num_classes, ignore_index=-1)
        self.val_ce = CrossEntropyLoss(ignore_index=-1)

    def forward(self, batch: BatchPair):
        x = batch[0]
        logits = self.model(x)
        return logits

    def loss(self, outputs: Any, batch: BatchPair):
        target = batch[1]
        loss = soft_cross_entropy(outputs, target, ignore_index=-1)  # type: ignore
        return loss

    def metrics(self, train: bool = False):
        metric_list = [self.train_miou, self.train_ce] if train else [self.val_miou, self.val_ce]
        return MetricCollection(metric_list)

    def validate(self, batch: BatchPair):
        assert self.training is False, "For validation, model must be in eval mode"
        target = batch[1]
        logits = self.forward(batch)
        return logits, target
