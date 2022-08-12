# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Single Shot Object Detection model with pretrained ResNet34 backbone extending :class:`.ComposerModel`."""

import os
import tempfile
from typing import Any, Sequence, Tuple, Union

import numpy as np
import requests
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from composer.models.base import ComposerModel
from composer.models.ssd.base_model import Loss
from composer.models.ssd.ssd300 import SSD300
from composer.models.ssd.utils import Encoder, SSDTransformer, dboxes300_coco
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['SSD']


class SSD(ComposerModel):
    """Single Shot Object detection Model with pretrained ResNet34 backbone extending :class:`.ComposerModel`.

    Args:
        input_size (int, optional): input image size. Default: ``300``.
        num_classes (int, optional): The number of classes to detect. Default: ``80``.
        overlap_threshold (float, optional): Minimum IOU threshold for NMS. Default: ``0.5``.
        nms_max_detections (int, optional): Max number of boxes after NMS. Default: ``200``.
        data (str, optional): path to coco dataset. Default: ``"/localdisk/coco"``.
    """

    def __init__(self, input_size: int, overlap_threshold: float, nms_max_detections: int, num_classes: int, data: str):
        super().__init__()

        self.input_size = input_size
        self.overlap_threshold = overlap_threshold
        self.nms_max_detections = nms_max_detections
        self.num_classes = num_classes
        url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        with tempfile.TemporaryDirectory() as tempdir:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                pretrained_backbone = os.path.join(tempdir, 'weights.pth')
                with open(pretrained_backbone, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                self.module = SSD300(self.num_classes, model_path=pretrained_backbone)

        dboxes = dboxes300_coco()
        self.loss_func = Loss(dboxes)

        self.encoder = Encoder(dboxes)
        self.data = data
        self.MAP = coco_map(self.data)
        val_annotate = os.path.join(self.data, 'annotations/instances_val2017.json')
        val_coco_root = os.path.join(self.data, 'val2017')
        input_size = self.input_size
        val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
        from composer.datasets.coco import COCODetection
        self.val_coco = COCODetection(val_coco_root, val_annotate, val_trans)

    def loss(self, outputs: Any, batch: Any) -> Union[Tensor, Sequence[Tensor]]:

        (_, _, _, bbox, label) = batch  #type: ignore
        if not isinstance(bbox, Tensor):
            raise TypeError('bbox must be a singular tensor')
        trans_bbox = bbox.transpose(1, 2).contiguous()

        ploc, plabel = outputs
        gloc, glabel = trans_bbox, label

        loss = self.loss_func(ploc, plabel, gloc, glabel)
        return loss

    def metrics(self, train: bool = False) -> Union[Metric, MetricCollection]:
        return self.MAP

    def forward(self, batch: Any) -> Tensor:
        (img, _, _, _, _) = batch  #type: ignore
        ploc, plabel = self.module(img)
        return ploc, plabel  #type: ignore

    def validate(self, batch: Any) -> Tuple[Any, Any]:
        inv_map = {v: k for k, v in self.val_coco.label_map.items()}
        ret = []
        overlap_threshold = self.overlap_threshold
        nms_max_detections = self.nms_max_detections

        (img, img_id, img_size, _, _) = batch  #type: ignore
        ploc, plabel = self.module(img)

        results = []
        try:
            results = self.encoder.decode_batch(ploc,
                                                plabel,
                                                overlap_threshold,
                                                nms_max_detections,
                                                nms_valid_thresh=0.05)
        except:
            print('No object detected')

        (htot, wtot) = [d.cpu().numpy() for d in img_size]  #type: ignore
        img_id = img_id.cpu().numpy()  #type: ignore
        if len(results) > 0:
            # Iterate over batch elements
            for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                loc, label, prob = [r.cpu().numpy() for r in result]  #type: ignore
                # Iterate over image detections
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id_, loc_[0]*wtot_, \
                                loc_[1]*htot_,
                                (loc_[2] - loc_[0])*wtot_,
                                (loc_[3] - loc_[1])*htot_,
                                prob_,
                                inv_map[label_]])

        return ret, ret


class coco_map(Metric):

    def __init__(self, data):
        super().__init__()
        try:
            from pycocotools.coco import COCO
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='coco',
                                                conda_channel='conda-forge',
                                                conda_package='pycocotools') from e
        self.add_state('predictions', default=[])
        val_annotate = os.path.join(data, 'annotations/instances_val2017.json')
        self.cocogt = COCO(annotation_file=val_annotate)

    def update(self, pred, target):
        self.predictions.append(pred)  #type: ignore
        np.squeeze(self.predictions)  #type: ignore

    def compute(self):
        try:
            from pycocotools.cocoeval import COCOeval
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='coco',
                                                conda_channel='conda-forge',
                                                conda_package='pycocotools') from e
        cocoDt = self.cocogt.loadRes(np.array(self.predictions))
        E = COCOeval(self.cocogt, cocoDt, iouType='bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
        return E.stats[0]
