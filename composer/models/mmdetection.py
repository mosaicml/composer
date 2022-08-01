# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper class that converts mmdet detection models to composer models"""

from __future__ import annotations

from typing import TYPCHECKING, List, Optional

from torchmetrics import Metric
from torchmetrics.collections import MetricCollection

from composer.models.base import ComposerModel

__all__ = ['MMDetModel']


class MMDetModel(ComposerModel):
    """
    A wrapper class that mmdetection detectors  to composer models.

    Args:
        model (mmdet.models.detectors.BaseDetector): # TODO
        metrics (list[Metric], optional): list of torchmetrics to apply to the output of `validate`. Default: ``None``.

    .. warning:: This wrapper is designed to work with mmdet datasets. #TODO convert to mmdet

    Example:

    .. testcode::

        from mmdet.models import build_model
        from mmcv import ConfigDict
        from composer.models import MMDetModel

        yolox_s_config = dict(
            type='YOLOX',
            input_size=(640, 640),
            random_size_range=(15, 25),
            random_size_interval=10,
            backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
            neck=dict(type='YOLOXPAFPN', in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
            bbox_head=dict(type='YOLOXHead', num_classes=num_classes, in_channels=128, feat_channels=128),
            train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
            test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

        yolox = build_model(ConfigDict(yolox_s_config))
        model = MMDetModel(yolox)
    """

    def __init__(self, model: mmdet.models.detectors.BaseDetector, metrics: Optional[List[Metric]] = None) -> None:
        super().__init__()
        self.model = model

        self.train_metrics = None
        self.valid_metrics = None

        if metrics:
            metric_collection = MetricCollection(metrics)
            self.train_metrics = metric_collection.clone(prefix='train_')
            self.valid_metrics = metric_collection.clone(prefix='val_')

    def forward(self, batch):
        return self.model(
            **batch)  # this will return a dictionary of 3 losses in train mode and model outputs in test mode

    def loss(self, outputs, batch, **kwargs):
        return outputs

    def validate(self, batch):
        # TODO model.forward can only take one image at a time in test mode...
        outputs = self.model(**batch)  # models behave differently in eval mode
        return outputs, batch

    def metrics(self, train: bool = False):
        return self.train_metrics if train else self.valid_metrics
