import os
import argparse
from typing import Any, Tuple

import numpy as np
from pycocotools.cocoeval import COCOeval
from torchmetrics import Metric

from composer.core.types import BatchPair, Metrics, Tensor, Tensors
from composer.models.base import ComposerModel
from composer.models.ssd.base_model import Loss
from composer.models.ssd.ssd300 import SSD300
from composer.models.ssd.utils import Encoder, SSDTransformer, dboxes300_coco


class SSD(ComposerModel):

    def __init__(self, input_size: int, overlap_threshold: float, nms_max_detections: int, pretrained_backbone: str,
                 num_classes: int):
        super().__init__()

        self.input_size = input_size
        self.overlap_threshold = overlap_threshold
        self.nms_max_detections = nms_max_detections
        self.pretrained_backbone = pretrained_backbone
        self.num_classes = num_classes
        self.module = SSD300(self.num_classes, model_path=self.pretrained_backbone)

        ##todo(laura): fix weights path
        dboxes = dboxes300_coco()
        self.loss_func = Loss(dboxes)
        self.MAP = coco_map()
        self.encoder = Encoder(dboxes)
        data = "/localdisk/coco" #self.datadir
        val_annotate = os.path.join(data, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(data, "val2017")
        input_size = self.input_size
        val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
        from composer.datasets.coco import COCODetection
        self.val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
        
    def loss(self, outputs: Any, batch: BatchPair) -> Tensors:

        (_, _, _, bbox, label) = batch
        trans_bbox = bbox.transpose(1, 2).contiguous()

        ploc, plabel = outputs
        gloc, glabel = trans_bbox, label

        loss = self.loss_func(ploc, plabel, gloc, glabel)
        return loss

    def metrics(self, train: bool = False) -> Metrics:
        return self.MAP

    def forward(self, batch: BatchPair) -> Tensor:
        (img, _, _, _, _) = batch
        ploc, plabel = self.module(img)

        return ploc, plabel

    def validate(self, batch: BatchPair) -> Tuple[Any, Any]:
        dboxes = dboxes300_coco()

        inv_map = {v: k for k, v in self.val_coco.label_map.items()}
        ret = []
        overlap_threshold = self.overlap_threshold
        nms_max_detections = self.nms_max_detections


        (img, img_id, img_size, _, _) = batch
        ploc, plabel = self.module(img.cuda())

        results = []
        try:
            results = self.encoder.decode_batch(ploc, plabel, overlap_threshold, nms_max_detections, nms_valid_thresh=0.05)
        except:
            print("No object detected")

        (htot, wtot) = [d.cpu().numpy() for d in img_size]
        img_id = img_id.cpu().numpy()
        if len(results) > 0:
            # Iterate over batch elements
            for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                loc, label, prob = [r.cpu().numpy() for r in result]
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

    def __init__(self):
        super().__init__()
        self.add_state("predictions", default=[])

    def update(self, pred, target):
        self.predictions.append(pred)
        np.squeeze(self.predictions)

    def compute(self):
        data = "/localdisk/coco" #self.datadir
        val_annotate = os.path.join(data, "annotations/instances_val2017.json")
        from composer.datasets.coco import COCO
        cocogt = COCO(annotation_file=val_annotate)
        cocoDt = cocogt.loadRes(np.array(self.predictions))
        E = COCOeval(cocogt, cocoDt, iouType='bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
        return E.stats[0]


def lr_warmup(optim, wb, iter_num, base_lr, args):
    if iter_num < wb:
        # mlperf warmup rule
        warmup_step = base_lr / (wb * (2**args.warmup_factor))
        new_lr = base_lr - (wb - iter_num) * warmup_step

        for param_group in optim.param_groups:
            param_group['lr'] = new_lr
