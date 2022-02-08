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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.detection.map import MeanAveragePrecision as MAP

from composer.core.types import BatchPair, Metrics, Tensor, Tensors
from composer.datasets.coco import COCO, COCODetection
from composer.models.base import ComposerModel
from composer.models.ssd.base_model import Loss
from composer.models.ssd.ssd300 import SSD300
from composer.models.ssd.ssd_hparams import SSDHparams
from composer.models.ssd.utils import DefaultBoxes, Encoder, SSDTransformer


class SSD(ComposerModel):

    def __init__(self, hparams: SSDHparams) -> None:
        super().__init__()

        self.hparams = hparams
        ln = COCODetection.labelnum
        self.module = SSD300(80, model_path="/mnt/r1z1/laura/composer/resnet34-333f7ec4.pth")
        ##todo(laura): fix weights path
        dboxes = dboxes300_coco()

        self.loss_func = Loss(dboxes)
        self.MAP = MAP()

    def loss(self, outputs: Any, batch: BatchPair) -> Tensors:

        (img, img_id, img_size, bbox, label) = batch
        trans_bbox = bbox.transpose(1, 2).contiguous()

        ploc, plabel = outputs
        gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                        Variable(label, requires_grad=False)

        loss = self.loss_func(ploc, plabel, gloc, glabel)
        return loss

    def metrics(self, train: bool = False) -> Metrics:

        return self.MAP

    def forward(self, batch: BatchPair) -> Tensor:
        (img, img_id, img_size, bbox, label) = batch
        context = contextlib.nullcontext if self.training else torch.no_grad

        img = Variable(img, requires_grad=True)
        ploc, plabel = self.module(img)

        return ploc, plabel

    def validate(self, batch: BatchPair) -> Tuple[Any, Any]:

        dboxes = dboxes300_coco()
        input_size = 300
        val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
        data = "/localdisk/coco"

        val_annotate = os.path.join(data, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(data, "val2017")
        train_annotate = os.path.join(data, "annotations/instances_train2017.json")
        train_coco_root = os.path.join(data, "train2017")
        val_coco = COCODetection(val_coco_root, val_annotate, val_trans)

        inv_map = {v: k for k, v in val_coco.label_map.items()}

        ret = []

        overlap_threshold = 0.50
        nms_max_detections = 200

        dboxes = dboxes300_coco()
        encoder = Encoder(dboxes)

        (img, img_id, img_size, _, _) = batch
        targets = get_boxes(val_annotate, img_id)

        with torch.no_grad():
            ploc, plabel = self.module(img)
            ploc, plabel = ploc.float(), plabel.float()

            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, 0.50, 200)[0]
                except:
                    # raise
                    print("")
                    print("No object detected in idx: {}".format(idx))
                    continue

                htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
                loc, label, prob = [r.cpu().numpy() for r in result]

                for loc_, label_, prob_ in zip(loc, label, prob):

                    ret.append([
                    dict(
                            boxes=torch.Tensor([[loc_[0] * wtot, \
                                                 loc_[1] * htot,
                                                 (loc_[2] - loc_[0]) * wtot,
                                                 (loc_[3] - loc_[1]) * htot]]),
                            scores=torch.Tensor([prob_]),
                            labels=torch.IntTensor([inv_map[label_]]),
                        )
                    ])

        print('lengths', len(ret), len(targets))
        #import pdb; pdb.set_trace()
        return ret, targets


def get_boxes(val_annotate, idss):
    ids = idss.tolist()
    from pycocotools.coco import COCO
    our_coco = COCO(annotation_file=val_annotate)
    import json
    json_file = "/localdisk//coco/annotations/instances_val2017.json"

    with open(json_file, 'r') as COCO:
        js = json.loads(COCO.read())
        anns_all = json.dumps(js['annotations'])

    annids = our_coco.getAnnIds(imgIds=ids)
    anns = our_coco.loadAnns(annids)

    t = []
    for ann in anns:
        x_topleft = ann['bbox'][0]
        y_topleft = ann['bbox'][1]
        bbox_width = ann['bbox'][2]
        bbox_height = ann['bbox'][3]

        cat = ann['category_id']

        t.append(dict(boxes=torch.Tensor([[x_topleft, y_topleft, bbox_width, bbox_height]]),
                      labels=torch.Tensor([cat])))

    #
    return t


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def lr_warmup(optim, wb, iter_num, base_lr, args):
    if iter_num < wb:
        # mlperf warmup rule
        warmup_step = base_lr / (wb * (2**args.warmup_factor))
        new_lr = base_lr - (wb - iter_num) * warmup_step

        for param_group in optim.param_groups:
            param_group['lr'] = new_lr
