# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core CopyPaste classes and functions."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple, Union

import os
from cv2 import resize
from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch import Tensor
from torch.nn import functional as F
import torchvision.transforms as T

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.loss.utils import check_for_index_targets


log = logging.getLogger(__name__)

__all__ = ['CopyPaste', 'copypaste_batch']


def _imshow(img):
    plt.figure()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def _imshow_single_ch(img):
    plt.figure()
    npimg = img.numpy()
    plt.imshow(npimg, cmap="gray")
    plt.show()


def copypaste_batch(input_dict, convert_to_binary_mask=True, max_copied_instances=None):
    """
    Randomly pastes objects onto an image.
    """

    output_dict = {
     "masks": [],
     "images": []
    }

    batch_size = len(input_dict["images"])

    while(len(output_dict["images"]) < batch_size):
        [i, j] = np.random.randint(0, high=batch_size, size=2)

        num_instances = input_dict["masks"][i].shape[0]
        num_copied_instances = random.randint(0, num_instances)
        if max_copied_instances is not None:
            num_copied_instances = min(num_copied_instances, max_copied_instances)


        rng = default_rng()
        src_instance_ids = rng.choice(num_instances, size=num_copied_instances, replace=False)

        trg_image = input_dict["images"][j]
        trg_masks = input_dict["masks"][j]

        for idx in range(num_copied_instances):
            trg_image, trg_masks = _copypaste_instance(input_dict, trg_image, trg_masks, i, j, src_instance_ids[idx], convert_to_binary_mask)    

        output_dict["images"].append(trg_image)
        output_dict["masks"].append(trg_masks)

    
    # fig_name = "SRC-"+input_dict["sample_names"][i]+"-TRG-"+input_dict["sample_names"][j]
    # _visualize_output_dict(output_dict, input_dict, batch_size, fig_name)


    return output_dict

def _copypaste_instance(input_dict, trg_image, trg_masks, i, j, src_instance_id, convert_to_binary_mask):
    src_masks = input_dict["masks"][i]
    src_instance_mask = src_masks[src_instance_id]
    src_image = input_dict["images"][i]

    if convert_to_binary_mask:
        src_instance_mask = torch.where(src_instance_mask==0, 0, 1)

    src_instance = torch.mul(src_image, src_instance_mask)

    src_instance, src_instance_mask = _jitter_instance(src_instance, src_instance_mask)

    trg_image = torch.where(src_instance_mask==0, trg_image, 0)
    trg_image = torch.clamp(trg_image + src_instance, min=0, max=1)


    for idx, trg_mask in enumerate(trg_masks):
        # TODO: here check if mask area is smaller than a threshold, disregard the mask
        trg_masks[idx] = torch.where(src_instance_mask==1, 0, trg_mask)

    new_trg_mask = torch.unsqueeze(src_instance_mask, dim=0)

    trg_masks = torch.cat((trg_masks, new_trg_mask), dim=0)

    # fig_name = "SRC-"+input_dict["sample_names"][i]+"-id-"+str(src_instance_id)+"-TRG-"+input_dict["sample_names"][j]
    # _visualize_copypaste(src_image, src_instance, input_dict["images"][j], trg_image, fig_name)

    return trg_image, trg_masks


def _jitter_instance(image, mask):
    padding_factor = 0.5

    jitter_seed = random.randint(0, 0xffff_ffff_ffff_ffff)

    torch.random.manual_seed(jitter_seed)
    random.seed(jitter_seed)

    trns = T.Compose([
        T.Pad(padding=(int(padding_factor * image.size(dim=1)), int(padding_factor * image.size(dim=2))), fill=0, padding_mode="constant"),
        T.RandomResizedCrop(size=(image.size(dim=1), image.size(dim=2)), scale=(0.01, 0.99), ratio=(1.0, 1.0)), 
        T.RandomHorizontalFlip(p=1.0)
        ])

    jittered_image = trns(image) 
    torch.random.manual_seed(jitter_seed)
    random.seed(jitter_seed)
    jittered_mask = trns(mask) 

    return jittered_image, jittered_mask

def _visualize_output_dict(output_dict, input_dict, batch_size, fig_name):
    dpi = 150

    fig, axarr = plt.subplots(2, batch_size, figsize=(18, 7), dpi=dpi)

    for col, image in enumerate(output_dict["images"]):

        ax = axarr[1, col]
        ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        _clean_axes(ax)

    for col, image in enumerate(input_dict["images"]):
        ax = axarr[0, col]
        ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        _clean_axes(ax)
        

    plt.suptitle("CopyPaste Augmentation", fontweight="bold")

    fig_out_path = os.path.join(".", "forks", "composer", "composer", "algorithms", "copypaste", "files", "out", "no_jittering", fig_name)
    plt.savefig(fig_out_path + ".png", dpi=dpi)


def _visualize_copypaste(src_image, src_instance, trg_image_before, trg_image_after, fig_name):
    
    dpi = 100
    img_1 = np.transpose(src_image.numpy(), (1, 2, 0))
    img_2 = np.transpose(src_instance.numpy(), (1, 2, 0))
    img_3 = np.transpose(trg_image_before.numpy(), (1, 2, 0))
    img_4 = np.transpose(trg_image_after.numpy(), (1, 2, 0))

    fig, axarr = plt.subplots(2, 2, figsize=(6, 6), dpi=dpi)
    
    ax = axarr[0, 0]
    ax.imshow(img_1)
    _clean_axes(ax)
    ax.set_title("Source Image")
    ax = axarr[0, 1]
    ax.imshow(img_2)
    _clean_axes(ax)
    ax.set_title("Source Instance")
    ax = axarr[1, 0]
    ax.imshow(img_3)
    _clean_axes(ax)
    ax.set_title("Target Image")
    ax = axarr[1, 1]
    ax.imshow(img_4)
    _clean_axes(ax)
    ax.set_title("Augmented Image")

    plt.suptitle("CopyPaste Augmentation", fontweight="bold")

    fig_out_path = os.path.join(".", "forks", "composer", "composer", "algorithms", "copypaste", "files", "out", "no_jittering", fig_name)
    plt.savefig(fig_out_path + ".png", dpi=dpi)

def _clean_axes(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.axis(('tight'))
    ax.set_aspect(aspect=1)


class CopyPaste(Algorithm):
    """
    Randomly pastes objects onto an image.
    """

    def __init__(
        self,
        input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1,
    ):
        self.input_key, self.target_key = input_key, target_key

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        input_dict = {
            "images": [],
            "masks": []
        }

        input_dict["images"] = state.batch_get_item(key=self.input_key)
        input_dict["masks"] = state.batch_get_item(key=self.target_key)

        augmented_dict = copypaste_batch(input_dict)  

        state.batch_set_item(key=self.input_key, value=augmented_dict)
        state.batch_set_item(key=self.target_key, value=augmented_dict)
