# Copyright 2021 MosaicML. All Rights Reserved.

from glob import glob
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torchvision
import yahp as hp
from torch.utils.data import Dataset

from composer.datasets.hparams import DataloaderSpec, DatasetHparams

from torchmetrics import Metric
import torch
import torch.nn as nn

@dataclass
class XBDDatasetHparams(DatasetHparams):
    is_train: bool = hp.required("whether to load the training or validation dataset")
    datadir: str = hp.required("data directory")
    download: bool = hp.required("whether to download the dataset, if needed")
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch", default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch", default=True)

    def initialize_object(self) -> DataloaderSpec:

        train_dataset, val_dataset = xBDTrainDataset(os.path.join(self.datadir, "train")), xBDValDataset(os.path.join(self.datadir, "test"))
        if self.is_train:
            return DataloaderSpec(
                dataset=train_dataset,
                drop_last=self.drop_last,
                shuffle=self.shuffle,
            )
        
        else:
            return DataloaderSpec(
                dataset=val_dataset,
                drop_last=self.drop_last,
                shuffle=self.shuffle,
            )
        

class F1(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("tp", default=torch.zeros((1,)), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros((1,)), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros((1,)), dist_reduce_fx="sum")

    def update(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        true_pos, false_neg, false_pos = self.get_stats(preds, targets, 1)
        self.tp[0] += true_pos
        self.fn[0] += false_neg
        self.fp[0] += false_pos

    def compute(self):
        return 200 * self.tp / (2 * self.tp + self.fp + self.fn)

    @staticmethod
    def get_stats(pred, targ, class_idx):
        true_pos = torch.logical_and(pred == class_idx, targ == class_idx).sum()
        false_neg = torch.logical_and(pred != class_idx, targ == class_idx).sum()
        false_pos = torch.logical_and(pred == class_idx, targ != class_idx).sum()
        return true_pos, false_neg, false_pos
    
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import numpy as np
import cv2

class xBDTrainDataset(Dataset):
    def __init__(self, path):
        self.imgs = sorted(glob(os.path.join(path, "images", f"*pre*")))
        self.lbls = sorted(glob(os.path.join(path, "targets", f"*pre*"))) 
        assert len(self.imgs) == len(self.lbls)
        self.zoom = A.RandomScale(p=0.2, scale_limit=(0, 0.3), interpolation=cv2.INTER_CUBIC)
        self.crop = A.CropNonEmptyMaskIfExists(p=1, width=512, height=512)
        self.hflip = A.HorizontalFlip(p=0.33)
        self.vflip = A.VerticalFlip(p=0.33)
        self.noise = A.GaussNoise(p=0.1)
        self.brctr = A.RandomBrightnessContrast(p=0.2)
        self.gamma = A.RandomGamma(p=0.2)
        self.normalize = A.Normalize()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, lbl = self.load_pair(idx)
        data = {"image": img, "mask": lbl}
        data = self.zoom(image=data["image"], mask=data["mask"])
        data = self.crop(image=data["image"], mask=data["mask"])
        data = self.hflip(image=data["image"], mask=data["mask"])
        data = self.vflip(image=data["image"], mask=data["mask"])
        img, lbl = data["image"], data["mask"]
        img = self.noise(image=img)["image"]
        img = self.brctr(image=img)["image"]
        img = self.gamma(image=img)["image"]
        img = self.normalize(image=img)["image"]
        lbl = np.expand_dims(lbl, 0)
        return {"image": np.transpose(img, (2, 0, 1)), "label": lbl}
    
    def load_pair(self, idx):
        img = cv2.imread(self.imgs[idx])
        lbl = cv2.imread(self.lbls[idx], cv2.IMREAD_UNCHANGED)
        return img, lbl
    
class xBDValDataset(Dataset):
    def __init__(self, path):
        self.imgs = sorted(glob(os.path.join(path, "images", f"*pre*")))
        self.lbls = sorted(glob(os.path.join(path, "targets", f"*pre*"))) 
        assert len(self.imgs) == len(self.lbls)
        self.normalize = A.Normalize()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, lbl = self.load_pair(idx)
        img = self.normalize(image=img)["image"]
        lbl = np.expand_dims(lbl, 0)
        return {"image": np.transpose(img, (2, 0, 1)), "label": lbl}
    
    def load_pair(self, idx):
        img = cv2.imread(self.imgs[idx])
        lbl = cv2.imread(self.lbls[idx], cv2.IMREAD_UNCHANGED)
        return img, lbl   
    
