import contextlib
from typing import Any

import torch
from torchmetrics.collections import MetricCollection
from torchvision.models import _utils, resnet
from torchvision.models.segmentation.deeplabv3 import ASPP, DeepLabV3

from composer.core.surgery import replace_module_classes
from composer.core.types import Batch, Tensor, Tensors
from composer.models.base import BaseMosaicModel
from composer.models.deeplabv3.deeplabv3_hparams import DeepLabv3Hparams
from composer.models.loss import CrossEntropyLoss, mIoU, soft_cross_entropy
from composer.models.model_hparams import Initializer


def deeplabv3_builder(backbone_arch: str, is_backbone_pretrained: bool, num_classes: int, sync_bn: bool):
    """Helper function to build a DeepLabV3 model"""

    # Instantiate backbone module
    if backbone_arch == 'resnet50':
        resnet.model_urls[backbone_arch] = "https://download.pytorch.org/models/resnet50-f46c3f97.pth"
    elif backbone_arch == 'resnet101':
        resnet.model_urls[backbone_arch] = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
    backbone = resnet.__dict__[backbone_arch](pretrained=is_backbone_pretrained,
                                              replace_stride_with_dilation=[False, True, True])
    backbone = _utils.IntermediateLayerGetter(backbone, return_layers={'layer4': 'out'})

    # Instantiate head module
    feat_extractor = ASPP(in_channels=2048, atrous_rates=[12, 24, 36], out_channels=256)
    feat_extractor.project = feat_extractor.project[:3]  # Remove dropout due to more variance in results
    classifier = torch.nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
    head = torch.nn.Sequential(feat_extractor, classifier)

    model = DeepLabV3(backbone, head, None)

    if sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model


class MosaicDeepLabV3(BaseMosaicModel):

    def __init__(self, hparams: DeepLabv3Hparams):
        super().__init__()
        self.hparams = hparams
        self.model = deeplabv3_builder(
            backbone_arch=self.hparams.backbone_arch,
            is_backbone_pretrained=self.hparams.is_backbone_pretrained,
            num_classes=self.hparams.num_classes,  # type: ignore
            sync_bn=self.hparams.sync_bn)

        # TODO: do I need initializer if pretrained is not specified?

    def forward(self, batch: Batch):
        x = batch[0]  # type: ignore

        context = contextlib.nullcontext if self.training else torch.no_grad  # is this necessary?

        with context():
            logits = self.model(x)['out']

        return logits

    def loss(self, outputs: Any, batch: Batch):
        _, target = batch

        loss = torch.nn.functional.cross_entropy(outputs, target, ignore_index=-1)
        return loss

    def metrics(self, train: bool = False):
        """

        """
        #return MetricCollection([mIoU(self.hparams.num_classes, self.hparams.ignore_index), CrossEntropyLoss()])
        return mIoU(self.hparams.num_classes, -1)

    def validate(self, batch: Batch):
        assert self.training is False, "For validation, model must be in eval mode"

        image, label = batch
        prediction = self.forward(batch).argmax(dim=1)

        return prediction, label
