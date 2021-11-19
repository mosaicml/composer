import torch

from .model import SSD300
from torch.autograd import Variable

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from .ssd_utils import Loss
from .coco import COCO
from .ssd_utils import dboxes300_coco, COCODetection, SSDTransformer, tencent_trick
from torch.optim.lr_scheduler import MultiStepLR
import torchmetrics

from torchmetrics.functional import average_precision




class Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = SSD300(self.hparams["initializers"])
        self.val_acc = torchmetrics.Accuracy()

    def _log_metrics(self, preds, y, loss, train=True):
        metric = self.val_acc
        [pred_image_indices, pred_probs, pred_labels, pred_bboxes] = preds
        [target_image_indices, target_labels, target_bboxes] = y

        metric(pred_image_indices, pred_probs, pred_labels, pred_bboxes,
               target_image_indices, target_labels, target_bboxes, 0.5, "COCO")

        self.log(f"acc/{'train' if train else 'val'}", metric)
        self.log(f"loss/{'train' if train else 'val'}", loss)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        learning_rate = self.hparams["lr"] * self.hparams["n_gpus"] * \
            (self.hparams["train_batch_size"] / 32)

        optimizer = torch.optim.SGD(
            tencent_trick(self.model),
            lr=learning_rate,
            momentum=self.hparams["momentum"],
            weight_decay=self.hparams["weight_decay"])

        scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=self.hparams["milestones"],
            gamma=0.1)

        return [optimizer], [scheduler]

    def warmup(optim, warmup_iters, iteration, base_lr):
        if iteration < warmup_iters:
            new_lr = 1. * base_lr / warmup_iters * iteration
            for param_group in optim.param_groups:
                param_group['lr'] = new_lr

    def optimizer_step(self, epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None):
        # TODO (laura): add back
        # self.warmup(optimizer, self.hparams["warmup"], iteration, self.hparams["lr"])
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def train_dataloader(self):
        default_boxes = dboxes300_coco()
        dataset = COCODetection(
            self.hparams["train_datadir"], self.hparams["train_instances"],
            SSDTransformer(default_boxes, self.hparams, (300, 300), val=False))

        train_sampler = None
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True, pin_memory=False,
            num_workers = self.hparams["workers_per_gpu"])

        return train_dataloader

    def val_dataloader(self):
        default_boxes = dboxes300_coco()
        dataset = COCODetection(
            self.hparams["val_datadir"], self.hparams["val_instances"],
            SSDTransformer(default_boxes, self.hparams, (300, 300), val=True))

        val_dataloader = DataLoader(
            dataset,
            batch_size=self.hparams["val_batch_size"],
            shuffle=False,
            num_workers = self.hparams["workers_per_gpu"])

        return val_dataloader

    def training_step(self, data, nbatch):
        dboxes = dboxes300_coco()
        loss_func = Loss(dboxes)

        (img, _, img_size, bbox, label) = data
        img = img.cuda()
        bbox = bbox.cuda()
        label = label.cuda()
        boxes_in_batch = len(label.nonzero())

        if boxes_in_batch != 0:
            ploc, plabel = self.model(img)
            ploc, plabel = ploc.float(), plabel.float()

            trans_bbox = bbox.transpose(1, 2).contiguous().cuda()

            label = label.cuda()
            gloc = Variable(trans_bbox, requires_grad=False)
            glabel = Variable(label, requires_grad=False)

            loss = loss_func(ploc, plabel, gloc, glabel)

        self.log(f"loss/{'train'}", loss)
        return {'loss': loss}

    def validation_step(self, data, nbatch):
        import numpy as np
        (img, _, img_size, bbox, label) = data
        ploc, plabel = self.model(img)
        

        def get_coco_ground_truth():
            import os
            val_annotate = os.path.join("/mnt/aws/datasets/coco/", "annotations/instances_val2017.json")
            cocoGt = COCO(annotation_file=val_annotate)
            return cocoGt
        
        cocoGt = get_coco_ground_truth()
        ret = []

        ## compute ACC
        for idx in range(ploc.shape[0]):
            # ease-of-use for specific predictions
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

        ret = np.array(ret).astype(np.float32)
        cocoDt = cocoGt.loadRes(ret)
    
        from pycocotools.cocoeval import COCOeval
        E = COCOeval(cocoGt, cocoDt, iouType='bbox')
        E.evaluate()
        E.accumulate()
        if args.local_rank == 0:
            E.summarize()
            print("Current AP: {:.5f}".format(E.stats[0]))
        else:
            # fix for cocoeval indiscriminate prints
            with redirect_stdout(io.StringIO()):
                E.summarize()


        self.log(f"acc/{'val'}", E.stats[0])
