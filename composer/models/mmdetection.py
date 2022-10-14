# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper class that converts mmdet detection models to composer models"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.collections import MetricCollection

from composer.models import ComposerModel

if TYPE_CHECKING:
    import mmdet

__all__ = ['MMDetModel']


class MMDetModel(ComposerModel):
    """A wrapper class that adapts mmdetection detectors to composer models.

    Args:
        model (mmdet.models.detectors.BaseDetector): An MMdetection Detector.
        metrics (list[Metric], optional): list of torchmetrics to apply to the output of `eval_forward`. Default: ``None``.

    .. warning:: This wrapper is designed to work with mmdet datasets.

    Example:

    .. code-block:: python

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
        yolox.init_weights()
        model = MMDetModel(yolox)
    """

    def __init__(
            self,
            model: mmdet.models.detectors.BaseDetector,  # type: ignore
            metrics: Optional[List[Metric]] = None) -> None:
        super().__init__()
        self.model = model

        self.train_metrics = None
        self.val_metrics = None

        if metrics:
            metric_collection = MetricCollection(metrics)
            self.train_metrics = metric_collection.clone(prefix='train_')
            self.val_metrics = metric_collection.clone(prefix='val_')

    def forward(self, batch):
        # this will return a dictionary of losses in train mode and model outputs in test mode.
        return self.model(**batch)

    def loss(self, outputs, batch, **kwargs):
        return outputs

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        """
        Args:
            batch (dict): a eval batch of the format:


            ``img`` (List[torch.Tensor]): list of image torch.Tensors of shape (batch, c, h , w).


            ``img_metas`` (List[Dict]): (1, batch_size) list of ``image_meta`` dicts.
        Returns: model predictions: A batch_size length list of dictionaries containg detection boxes in (x,y, x2, y2) format, class labels, and class probabilities.
        """
        device = batch['img'][0].device
        batch.pop('gt_labels')
        batch.pop('gt_bboxes')
        results = self.model(return_loss=False, rescale=True, **batch)  # models behave differently in eval mode

        # outputs are a list of bbox results (x, y, x2, y2, score)
        # pack mmdet bounding boxes and labels into the format for torchmetrics MAP expects
        preds = []
        for bbox_result in results:
            boxes_scores = np.vstack(bbox_result)
            boxes, scores = torch.from_numpy(boxes_scores[..., :-1]).to(device), torch.from_numpy(
                boxes_scores[..., -1]).to(device)
            labels = [np.full(result.shape[0], i, dtype=np.int32) for i, result in enumerate(bbox_result)]
            pred = {
                'labels': torch.from_numpy(np.concatenate(labels)).to(device).long(),
                'boxes': boxes.float(),
                'scores': scores.float()
            }
            preds.append(pred)
        return preds

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics
        return metrics if metrics else {}

    def update_metric(self, batch: Any, outputs: Any, metric: Metric):
        targets_box = batch.pop('gt_bboxes')[0]
        targets_cls = batch.pop('gt_labels')[0]
        targets = []
        for i in range(len(targets_box)):
            t = {'boxes': targets_box[i], 'labels': targets_cls[i]}
            targets.append(t)
        metric.update(outputs, targets)
