import logging
import os
import random
import time
from argparse import ArgumentParser
from typing import Any, Optional, Tuple
import numpy as np
import torch
from composer.models.ssd.base_model import Loss
from composer.models.ssd.ssd300 import SSD300
from torch.autograd import Variable
from torch.utils.data import DataLoader

from composer.models.ssd.utils import  DefaultBoxes, Encoder, SSDTransformer
from composer.datasets.coco import COCODetection
from composer.core.types import BatchPair, Metrics, Tensor, Tensors
from composer.models.base import BaseMosaicModel
from torchmetrics.classification.accuracy import Accuracy
from composer.models.ssd.ssd_hparams import SSDHparams
from composer.models.ssd.ssd300 import SSD300
from PIL import Image
import contextlib

_BASE_LR = 2.5e-3


class SSD(BaseMosaicModel):
    def __init__(self, hparams: SSDHparams) -> None:
        super().__init__()

        self.hparams = hparams
        ln = COCODetection.labelnum
        self.module = SSD300(80, model_path="/mnt/cota/laura/composer/composer/models/ssd/resnet34-333f7ec4.pth")#args.pretrained_backbone)
        dboxes = dboxes300_coco()
        
        self.loss_func = Loss(dboxes)
        #self.mAP = mAP()

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
        return Accuracy()#self.mAP

    def forward(self, batch: BatchPair) -> Tensor:
        (img, img_id, img_size, bbox, label) = batch
        context = contextlib.nullcontext if self.training else torch.no_grad

        img = Variable(img, requires_grad=True)
        ploc, plabel = self.module(img)

        return ploc, plabel
        
    def validate(self, batch: BatchPair) -> Tuple[Any, Any]:
        
        #val_coco = COCODetection(val_coco_root, val_annotate, val_trans)

        #inv_map = {v: k for k, v in batch.label_map.items()}
        
        (img, img_id, img_size, bbox, label) = batch
        ret = []

        overlap_threshold = 0.50
        nms_max_detections = 200

        ploc, plabel = self.module(img)
        dboxes = dboxes300_coco()

        encoder = Encoder(dboxes)

        with torch.no_grad():

            results = encoder.decode_batch(ploc,
                                           plabel,
                                           overlap_threshold,
                                           nms_max_detections,
                                           nms_valid_thresh=0.05)
            
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

    
        return ret


from torchmetrics import Metric

def mAP(Metric):
    
    
    cocoDt = cocoGt.loadRes(np.array(ret))

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()

    current_accuracy = E.stats[0]

    return current_accuracy# >= threshold
        



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

