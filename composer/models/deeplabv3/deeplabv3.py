import contextlib
from typing import Any

import torch
from torchvision.models.segmentation import deeplabv3_resnet50

from composer.core.types import BatchPair, Tensor, Tensors
from composer.models.base import BaseMosaicModel
from composer.models.deeplabv3.deeplabv3_hparams import DeepLabv3Hparams
from composer.models.loss import CrossEntropyLoss, mIoU, soft_cross_entropy


class DeepLabv3(BaseMosaicModel):

    def __init__(self, hparams: DeepLabv3Hparams):
        super().__init__()
        self.hparams = hparams
        self.model = deeplabv3_resnet50(self.hparams.is_pretrained,
                                        progress=False,
                                        num_classes=self.hparams.num_classes)

    def forward(self, batch: BatchPair
               ):  # Should the forward pass take a batch pair? We shouldn't expect the forward pass to have labels
        x, _ = batch

        context = contextlib.nullcontext if self.training else torch.no_grad  # should this be added?

        with context():
            logits = self.model(x)['out']

        return logits

    def loss(self, outputs: Any, batch: BatchPair):
        _, target = batch
        #print(outputs.shape, target.shape, torch.unique(target))
        loss = soft_cross_entropy(outputs, target)
        return loss

    def metrics(self, train: bool = False):
        """

        """
        return mIoU(self.hparams.num_classes, self.hparams.ignore_index)

    def validate(self, batch: BatchPair):
        assert self.training is False, "For validation, model must be in eval mode"

        image, label = batch
        prediction = self.forward(batch).argmax(dim=1)

        return prediction, label
