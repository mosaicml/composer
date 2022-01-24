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

#from torchmetrics.detection.map. import MAP
#from torchmetrics.detection.map import MeanAveragePrecision as MAP
from composer.core.types import BatchPair, Metrics, Tensor, Tensors
from composer.datasets.coco import COCO, COCODetection
from composer.models.base import BaseMosaicModel
from composer.models.ssd.base_model import Loss
from composer.models.ssd.ssd300 import SSD300
from composer.models.ssd.ssd_hparams import SSDHparams
from composer.models.ssd.utils import DefaultBoxes, Encoder, SSDTransformer


class SSD(BaseMosaicModel):

    def __init__(self, hparams: SSDHparams) -> None:
        super().__init__()

        self.hparams = hparams
        ln = COCODetection.labelnum
        self.module = SSD300(
            80,
            model_path="/mnt/cota/laura/composer/composer/models/ssd/resnet34-333f7ec4.pth")
        dboxes = dboxes300_coco()

        self.loss_func = Loss(dboxes)
        self.MAP = my_map()

    def loss(self, outputs: Any, batch: BatchPair) -> Tensors:

        (img, img_id, img_size, bbox, label) = batch
        trans_bbox = bbox.transpose(1, 2).contiguous()

        ploc, plabel = outputs
        gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                        Variable(label, requires_grad=False)

        loss = self.loss_func(ploc, plabel, gloc, glabel)
        '''
        if not np.isinf(loss.item()):
            avg_loss = 0.999 * avg_loss + 0.001 * loss.item()
        '''
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
        data = "/mnt/cota/datasets/coco"

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

        '''
        val_coco_dl = DataLoader(val_coco,
                                    batch_size=32,
                                    shuffle=False,
                                    sampler=None,
                                    num_workers=8)

        for nbatch, (img, img_id, img_size, _, _) in enumerate(val_coco_dl):
        '''
        (img, img_id, img_size, _, _) = batch
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
                    ret.append([img_id[idx], loc_[0] * wtot, \
                                loc_[1] * htot,
                                (loc_[2] - loc_[0]) * wtot,
                                (loc_[3] - loc_[1]) * htot,
                                prob_,
                                inv_map[label_]])

        #ret = np.array(ret).astype(np.float32)

        final_results = ret

        return ret, ret


class my_map(Metric):
    
    def __init__(self):
        super().__init__(dist_sync_on_step=True)
        self.add_state("n_updates", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, pred, target):
        self.n_updates += 1
        data = "/mnt/cota/datasets/coco"
        val_annotate = os.path.join(data, "annotations/instances_val2017.json")
        
        self.cocogt = COCO(annotation_file=val_annotate)

        
        
    def compute(self, pred):
        self.cocodt = self.cocogt.loadRes(pred)
        E = COCOeval(self.cocogt, self.cocodt, iouType='bbox')
        print('here')
        E.evaluate()
        E.accumulate()
        E.summarize()

        return E.stats[0]

        

def transform_d(val_annotate):
    from pycocotools.coco import COCO
    our_coco = COCO(annotation_file=val_annotate)
    import json
    json_file = "/mnt/cota/datasets/coco/annotations/instances_val2017.json"
    with open(json_file, 'r') as COCO:
        js = json.loads(COCO.read())
        cat_names = json.dumps(js['categories'])
    cat_ids = our_coco.getCatIds(catNms=cat_names)
    target = []
    for cat_id in cat_ids:
        # get annotations for a specific class
        ann_ids = our_coco.getAnnIds(catIds=cat_id)
        anns = our_coco.loadAnns(ann_ids)

        for ann in anns:
            x_topleft = ann['bbox'][0]
            y_topleft = ann['bbox'][1]
            bbox_width = ann['bbox'][2]
            bbox_height = ann['bbox'][3]

            img_id = ann['image_id']
            target.append(
                dict(boxes=torch.Tensor([[x_topleft, y_topleft, bbox_width, bbox_height]]),
                     labels=torch.Tensor([img_id])))

    return target
    #create sequence


from torchmetrics import Metric


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


#torch.backends.cudnn.benchmark = True
