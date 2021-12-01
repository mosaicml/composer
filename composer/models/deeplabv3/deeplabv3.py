import contextlib
from typing import Any

import torch
from torchmetrics.collections import MetricCollection
from torchvision.models import resnet
from torchvision.models.segmentation import deeplabv3_resnet50

from composer.core.surgery import replace_module_classes
from composer.core.types import BatchPair, Tensor, Tensors
from composer.models.base import BaseMosaicModel
from composer.models.deeplabv3.deeplabv3_hparams import DeepLabv3Hparams
from composer.models.loss import CrossEntropyLoss, mIoU, soft_cross_entropy
from composer.models.model_hparams import Initializer


class DeepLabv3(BaseMosaicModel):

    def __init__(self, hparams: DeepLabv3Hparams):
        super().__init__()
        self.hparams = hparams
        self.model = deeplabv3_resnet50(False,
                                        progress=False,
                                        num_classes=self.hparams.num_classes,
                                        aux_loss=False,
                                        pretrained_backbone=self.hparams.is_pretrained == 'old')
        # replace 1x1 conv with 3x3 conv
        conv_to_replace = self.model.classifier[0].project[0]
        if self.hparams.penult_kernel == 3:
            self.model.classifier[0].project[0] = torch.nn.Conv2d(conv_to_replace.in_channels,
                                                                  conv_to_replace.out_channels,
                                                                  kernel_size=3,
                                                                  stride=1,
                                                                  padding=1,
                                                                  bias=False)
            torch.nn.init.kaiming_normal_(self.model.classifier[0].project[0].weight)
        # Change dropout
        if self.hparams.dropout:
            self.model.classifier[0].project[3] = torch.nn.Dropout2d(self.hparams.dropout)
        else:
            self.model.classifier[0].project[3] = torch.nn.Identity()
        # remove 3x3 conv
        self.model.classifier[1] = torch.nn.Identity()
        self.model.classifier[2] = torch.nn.Identity()
        self.model.classifier[3] = torch.nn.Identity()
        if self.hparams.is_pretrained == 'new':
            resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-f46c3f97.pth"
            backbone = resnet.resnet50(pretrained=True)
            del backbone.fc
            self.model.backbone.load_state_dict(backbone.state_dict())
        if self.hparams.sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if not self.hparams.is_pretrained:
            for initializer in self.hparams.initializers:
                initializer = Initializer(initializer)
                self.model.apply(initializer.get_initializer())

    # Should the forward pass take a batch pair? We shouldn't expect the forward pass to have labels
    def forward(self, batch: BatchPair):
        x, _ = batch

        context = contextlib.nullcontext if self.training else torch.no_grad  # should this be added?

        with context():
            logits = self.model(x)['out']

        return logits

    def loss(self, outputs: Any, batch: BatchPair):
        _, target = batch

        loss = torch.nn.functional.cross_entropy(outputs, target, ignore_index=-1)
        return loss

    def metrics(self, train: bool = False):
        """

        """
        #return MetricCollection([mIoU(self.hparams.num_classes, self.hparams.ignore_index), CrossEntropyLoss()])
        return mIoU(self.hparams.num_classes, self.hparams.ignore_index)

    def validate(self, batch: BatchPair):
        assert self.training is False, "For validation, model must be in eval mode"

        image, label = batch
        prediction = self.forward(batch).argmax(dim=1)

        return prediction, label
