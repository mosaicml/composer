# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Any, List

import torch
from torchmetrics.collections import MetricCollection
from torchvision.models import _utils, resnet
from torchvision.models.segmentation.deeplabv3 import ASPP, DeepLabV3

from composer.core.types import BatchPair
from composer.models.base import ComposerModel
from composer.models.loss import CrossEntropyLoss, MIoU, soft_cross_entropy
from composer.models.model_hparams import Initializer


def deeplabv3_builder(num_classes: int,
                      backbone_arch: str = 'resnet101',
                      is_backbone_pretrained: bool = True,
                      sync_bn: bool = True,
                      initializers: List[Initializer] = []):
    """Helper function to build a torchvision DeepLabV3 model with a 3x3 convolution layer and dropout removed.

    Args:
        num_classes (int): number of classes in the segmentation task.
        backbone_arch (str): the architecture to use for the backbone. Must be either ['resnet50', 'resnet101'].
            Default is 'resnet101'.
        is_backbone_pretrained (bool): if true (default), use pretrained weights for the backbone.
        sync_bn (bool): if true (default), replace all BatchNorm layers with SyncBatchNorm layers.
    """

    # Instantiate backbone module
    if backbone_arch == 'resnet50':
        resnet.model_urls[backbone_arch] = "https://download.pytorch.org/models/resnet50-f46c3f97.pth"
    elif backbone_arch == 'resnet101':
        resnet.model_urls[backbone_arch] = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
    else:
        raise ValueError(f"backbone_arch must be one of ['resnet50', 'resnet101'] not {backbone_arch}")
    backbone = getattr(resnet, backbone_arch)(pretrained=is_backbone_pretrained,
                                              replace_stride_with_dilation=[False, True, True])
    backbone = _utils.IntermediateLayerGetter(backbone, return_layers={'layer4': 'out'})

    # Instantiate head module
    feat_extractor = ASPP(in_channels=2048, atrous_rates=[12, 24, 36], out_channels=256)
    feat_extractor.project = feat_extractor.project[:3]  # Remove dropout due to higher standard deviation
    classifier = torch.nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
    head = torch.nn.Sequential(feat_extractor, classifier)

    model = DeepLabV3(backbone, head, aux_classifier=None)

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
    """DeepLabV3 model extending the :class:`ComposerClassifier`.

    See `<arxiv.org/abs/1706.05587>`_ for more details on the DeepLabV3 architecture.

    Args:
        num_classes (int): the number of classes in the segmentation task.
        backbone_arch (str): the backbone architecture to use, either 'resnet50', 'resnet101'. Default is 'resnet101'.
        is_backbone_pretrained (bool): if true (default), use pre-trained weights for backbone.
        sync_bn (bool): if true (default), use SyncBatchNorm to sync batch norm statistics across GPUs.
    """

    def __init__(self,
                 num_classes: int,
                 backbone_arch: str = 'resnet101',
                 is_backbone_pretrained: bool = True,
                 sync_bn: bool = True,
                 initializers: List[Initializer] = []):

        super().__init__()
        self.num_classes = num_classes
        self.model = deeplabv3_builder(backbone_arch=backbone_arch,
                                       is_backbone_pretrained=is_backbone_pretrained,
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
        logits = self.model(x)['out']
        return logits

    def loss(self, outputs: Any, batch: BatchPair):
        """Calculate the specified loss for training."""
        target = batch[1]
        loss = soft_cross_entropy(outputs, target, ignore_index=-1)  # type: ignore
        return loss

    def metrics(self, train: bool = False):
        """Metrics to compute during validation."""
        metric_list = [self.train_miou, self.train_ce] if train else [self.val_miou, self.val_ce]
        return MetricCollection(metric_list)

    def validate(self, batch: BatchPair):
        """Generate outputs used during validation."""
        assert self.training is False, "For validation, model must be in eval mode"
        target = batch[1]
        logits = self.forward(batch)
        return logits, target
