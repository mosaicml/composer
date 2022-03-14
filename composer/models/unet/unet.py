# Copyright 2021 MosaicML. All Rights Reserved.

"""A U-Net model extending :class:`.ComposerModel`."""

import logging
import textwrap
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from composer.core.types import BatchPair, Metrics, Tensor, Tensors
from composer.models.base import ComposerModel
from composer.models.loss import Dice
from composer.models.unet.model import UNet as UNetModel

log = logging.getLogger(__name__)

__all__ = ["UNet"]


class UNet(ComposerModel):
    """A U-Net model extending :class:`.ComposerModel`.

    See U-Net: Convolutional Networks for Biomedical Image Segmentation (`Ronneberger et al, 2015`_)
    on the U-Net architecture.

    Args:
        num_classes (int): The number of classes. Needed for classification tasks. Default: ``3``.

    .. _Ronneberger et al, 2015: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, num_classes: Optional[int] = 3) -> None:
        super().__init__()
        try:
            from monai.losses import DiceLoss
        except ImportError as e:
            raise ImportError(
                textwrap.dedent("""\
                Composer was installed without unet support. To use timm with Composer, run `pip install mosaicml[unet]`
                if using pip or `conda install -c conda-forge monai` if using Anaconda.""")) from e

        self.module = self.build_nnunet()

        self.dice = Dice(num_classes=num_classes)
        self.dloss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, batch=True)
        self.closs = nn.CrossEntropyLoss()

    def loss(self, outputs: Any, batch: BatchPair, *args, **kwargs) -> Tensors:
        _, y = batch
        y = y.squeeze(1)  # type: ignore
        loss = self.dloss(outputs, y)
        loss += self.closs(outputs, y[:, 0].long())
        return loss

    @staticmethod
    def metric_mean(name, outputs):
        return torch.stack([out[name] for out in outputs]).mean(dim=0)

    def metrics(self, train: bool = False) -> Metrics:
        return self.dice

    def forward(self, batch: BatchPair) -> Tensor:
        x, _ = batch
        x = x.squeeze(1)  # type: ignore
        logits = self.module(x)
        return logits

    def inference2d(self, image):
        """Runs inference on a 3D image, by passing each depth slice through the model."""
        batch_modulo = image.shape[2] % 64
        if batch_modulo != 0:
            batch_pad = 64 - batch_modulo
            image = nn.ConstantPad3d((0, 0, 0, 0, batch_pad, 0), 0)(image)

        image = torch.transpose(image.squeeze(0), 0, 1)
        preds_shape = (image.shape[0], 4, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for start in range(0, image.shape[0] - 64 + 1, 64):
            end = start + 64
            with torch.no_grad():
                pred = self.module(image[start:end])
            preds[start:end] = pred.data
        if batch_modulo != 0:
            preds = preds[batch_pad:]  # type: ignore
        return torch.transpose(preds, 0, 1).unsqueeze(0)

    def validate(self, batch: BatchPair) -> Tuple[Any, Any]:
        assert self.training is False, "For validation, model must be in eval mode"
        image, target = batch
        pred = self.inference2d(image)
        return pred, target[:, 0].long()  # type: ignore

    def build_nnunet(self) -> torch.nn.Module:
        kernels = [[3, 3]] * 6
        strides = [[1, 1]] + [[2, 2]] * 5
        model = UNetModel(in_channels=4,
                          n_class=4,
                          kernels=kernels,
                          strides=strides,
                          dimension=2,
                          residual=True,
                          normalization_layer="batch",
                          negative_slope=0.01)

        return model
