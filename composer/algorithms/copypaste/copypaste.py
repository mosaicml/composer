# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core CopyPaste classes and functions."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple, Union

import os
from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torchvision
from torch import Tensor
from torch.nn import functional as F


from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.loss.utils import check_for_index_targets


log = logging.getLogger(__name__)

__all__ = ['CopyPaste', 'copypaste_batch']


def _imshow(img):
    plt.figure()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()

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


    batch_size = len(input_dict["sample_names"])

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

        # print("one done")
    
    # fig_name = "SRC-"+input_dict["sample_names"][i]+"-TRG-"+input_dict["sample_names"][j]
    # _visualize_output_dict(output_dict, input_dict, batch_size, fig_name)


    return output_dict

def _copypaste_instance(input_dict, trg_image, trg_masks, i, j, src_instance_id, convert_to_binary_mask):
    # src_instance_id = 3
    src_masks = input_dict["masks"][i]
    src_instance_mask = src_masks[src_instance_id]
    src_image = input_dict["images"][i]

    if convert_to_binary_mask:
        src_instance_mask = torch.where(src_instance_mask==0, 0, 1)

    src_instance = torch.mul(src_image, src_instance_mask)
    trg_image = torch.where(src_instance_mask==0, trg_image, 0)
    # _imshow(trg_image)
    trg_image = trg_image + src_instance
    # _imshow(trg_image)


    for idx, trg_mask in enumerate(trg_masks):
        # _imshow(src_instance_mask*255)
        # _imshow(trg_mask)
        temp_mask = torch.where(src_instance_mask==1, 0, trg_mask)
        trg_masks[idx] = temp_mask
        # _imshow(temp_mask)

        # print(i)

    new_trg_mask = torch.unsqueeze(src_instance_mask, dim=0)

    trg_masks = torch.cat((trg_masks, new_trg_mask), dim=0)






    fig_name = "SRC-"+input_dict["sample_names"][i]+"-id-"+str(src_instance_id)+"-TRG-"+input_dict["sample_names"][j]
    # _visualize_copypaste(src_image, src_instance, input_dict["images"][j], trg_image, fig_name)

    return trg_image, trg_masks

def _visualize_output_dict(output_dict, input_dict, batch_size, fig_name):
    
    dpi = 150
    # img_1 = np.transpose(src_image.numpy(), (1, 2, 0))
    # img_2 = np.transpose(src_instance.numpy(), (1, 2, 0))
    # img_3 = np.transpose(trg_image_before.numpy(), (1, 2, 0))
    # img_4 = np.transpose(trg_image_after.numpy(), (1, 2, 0))

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
    # plt.show()

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
    # plt.show()

    fig_out_path = os.path.join(".", "forks", "composer", "composer", "algorithms", "copypaste", "files", "out", "no_jittering", fig_name)
    plt.savefig(fig_out_path + ".png", dpi=dpi)

def _clean_axes(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axes.yaxis.set_visible(False)
    # ax.set_xlim(200, 340)
    # ax.set_ylim(0, 8)
    ax.axes.xaxis.set_visible(False)
    # ax.axis(('auto'))
    ax.axis(('tight'))
    ax.set_aspect(aspect=1)

def _decompose_mask(mask: Tensor):
    """
    decomposes a color-coded mask tensor into an array of individual instance masks
    """
    mask_npy = mask.numpy()
    """
    0  4
    1  2

    0  3
    2  3

    0  2
    1  2
    
    """
    # mask_npy = np.array([[[0, 4], [1, 2]], [[0, 3], [2, 3]], [[0, 2], [1, 2]]])


    # plt.imshow(np.transpose(mask_npy, (1, 2, 0)))
    

    # mask_npy_flatten = mask_npy.reshape(-1, mask_npy.shape[0])
    # mask_npy_flatten = mask_npy.reshape(2, 2, 3)
    mask_npy_moved = np.moveaxis(mask_npy, 0, -1)
    plt.imshow(mask_npy_moved)
    mask_npy_flatten = mask_npy_moved.reshape(-1, 3)

    mask_ids = np.unique(mask_npy_flatten, axis=0)

    # mask_ids = np.unique(mask_npy_flatten, axis=1)
    
    # print(mask_ids)
    for id in mask_ids:
        print(255*id)
    return

class CopyPaste(Algorithm):
    """
    Randomly pastes objects onto an image.
    """

    def __init__(
        self,
        num_classes: int,
        uniform_sampling: bool = False,
        input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1,
    ):
        self.num_classes = num_classes
        self._uniform_sampling = uniform_sampling
        self._indices = torch.Tensor()
        self._cutmix_lambda = 0.0
        self._bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.input_key, self.target_key = input_key, target_key

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        input = state.batch_get_item(key=self.input_key)
        target = state.batch_get_item(key=self.target_key)

        assert isinstance(input, Tensor) and isinstance(target, Tensor), \
            'Multiple tensors for inputs or targets not supported yet.'

        new_input, new_target = copypaste_batch(
            input=input,
            target=target,
            num_classes=self.num_classes,
            bbox=self._bbox,
            indices=self._indices,
        )

        state.batch_set_item(key=self.input_key, value=new_input)
        state.batch_set_item(key=self.target_key, value=new_target)
