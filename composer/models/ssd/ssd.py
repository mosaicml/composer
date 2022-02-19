import contextlib
import logging
import os
import random
import time
from argparse import ArgumentParser
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchmetrics import Metric

from composer.core.types import BatchPair, Metrics, Tensor, Tensors

from composer.models.base import ComposerModel
from composer.models.ssd.base_model import Loss
from composer.models.ssd.ssd300 import SSD300
from composer.models.ssd.ssd_hparams import SSDHparams
from composer.models.ssd.utils import DefaultBoxes, Encoder, SSDTransformer, dboxes300_coco




class SSD(ComposerModel):

    def __init__(self, hparams: SSDHparams) -> None:
        super().__init__()

        self.hparams = hparams
        self.module = SSD300(80, model_path="/mnt/r1z1/laura/composer/resnet34-333f7ec4.pth")
        ##todo(laura): fix weights path
        dboxes = dboxes300_coco()
        self.loss_func = Loss(dboxes)
        self.MAP = my_map()

    def loss(self, outputs: Any, batch: BatchPair) -> Tensors:

        (img, img_id, img_size, bbox, label) = batch
        trans_bbox = bbox.transpose(1, 2).contiguous()

        ploc, plabel = outputs
        gloc, glabel = trans_bbox, label

        loss = self.loss_func(ploc, plabel, gloc, glabel)
        return loss

    def metrics(self, train: bool = False) -> Metrics:
        return self.MAP

    def forward(self, batch: BatchPair) -> Tensor:
        (img, img_id, img_size, bbox, label) = batch
        ploc, plabel = self.module(img)

        return ploc, plabel

    def validate(self, batch: BatchPair) -> Tuple[Any, Any]:
        dboxes = dboxes300_coco()
        data = "/localdisk/coco"
        val_annotate = os.path.join(data, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(data, "val2017")
        input_size = 300
        from composer.datasets.coco import COCO, COCODetection        
        val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)

        val_coco = COCODetection(val_coco_root, val_annotate, val_trans)

        inv_map = {v: k for k, v in val_coco.label_map.items()}
        ret = []
        overlap_threshold = 0.50
        nms_max_detections = 200
        encoder = Encoder(dboxes)

        (img, img_id, img_size, _, _) = batch
        ploc, plabel = self.module(img.cuda())

        try:
            results = encoder.decode_batch(ploc, plabel, overlap_threshold, nms_max_detections, nms_valid_thresh=0.05)
        except:
            print("No object detected in batch: {}".format(nbatch))

        (htot, wtot) = [d.cpu().numpy() for d in img_size]
        img_id = img_id.cpu().numpy()
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


class my_map(Metric):

    def __init__(self):#, dist_sync_on_step=True):
        super().__init__()
        self.add_state("predictions", default=[])

    def update(self, pred, target):
        #for i in len(
        #import pdb; pdb.set_trace()
        self.predictions.append(pred)
        np.squeeze(self.predictions)


    def compute(self):
        #import pdb; 
        data = "/localdisk/coco"
        val_annotate = os.path.join(data, "annotations/instances_val2017.json")
        from composer.datasets.coco import COCO
        cocogt = COCO(annotation_file=val_annotate)
        #import pdb; pdb.set_trace()
        
        cocoDt = cocogt.loadRes(np.array(self.predictions))#np.squeeze(self.predictions))

        E = COCOeval(cocogt, cocoDt, iouType='bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
        print('acc', E.stats[0])
        return E.stats[0]


def lr_warmup(optim, wb, iter_num, base_lr, args):
    if iter_num < wb:
        # mlperf warmup rule
        warmup_step = base_lr / (wb * (2**args.warmup_factor))
        new_lr = base_lr - (wb - iter_num) * warmup_step

        for param_group in optim.param_groups:
            param_group['lr'] = new_lr
