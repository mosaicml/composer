import contextlib
from typing import Any

import torch
from torchmetrics.collections import MetricCollection
from torchvision.models import resnet50
from torchvision.models.segmentation import deeplabv3_resnet50

from composer.core.surgery import replace_module_classes
from composer.core.types import BatchPair, Tensor, Tensors
from composer.models.base import BaseMosaicModel
from composer.models.deeplabv3.deeplabv3_hparams import DeepLabv3Hparams
from composer.models.loss import CrossEntropyLoss, mIoU, soft_cross_entropy


def bn_to_syncbn(module, module_index):
    new_module = torch.nn.SyncBatchNorm(num_features=module.num_features,
                                        eps=module.eps,
                                        momentum=module.momentum,
                                        affine=module.affine,
                                        track_running_stats=module.track_running_stats)
    new_module.running_mean = module.running_mean
    new_module.running_var = module.running_var

    if module.affine:
        new_module.weight.data = module.weight.data.clone().detach()
        new_module.bias.data = module.bias.data.clone().detach()

    return new_module


class DeepLabv3(BaseMosaicModel):

    def __init__(self, hparams: DeepLabv3Hparams):
        super().__init__()
        self.hparams = hparams
        self.model = deeplabv3_resnet50(False, progress=False, num_classes=self.hparams.num_classes, aux_loss=False)
        if self.hparams.is_pretrained:
            backbone = resnet50(pretrained=True, progress=False)
            del backbone.fc
            self.model.backbone.load_state_dict(backbone.state_dict())
        if self.hparams.sync_bn:
            policy = {torch.nn.BatchNorm2d: bn_to_syncbn}
            replace_module_classes(self.model, policy)

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
