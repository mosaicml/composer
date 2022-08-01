# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core CopyPaste classes and functions.

Notes:
- add thresholding for small masks
"""

from __future__ import annotations

import logging
from cv2 import transform
import matplotlib.pyplot as plt
import os
from typing import Any, Callable, Optional, Tuple, Union

from numpy.random import default_rng
import numpy as np
import random
import torch
import torchvision.transforms as T

from composer.core import Algorithm, Event, State
from composer.loggers import Logger

MAX_TORCH_SEED = 0xffff_ffff_ffff_ffff
log = logging.getLogger(__name__)

__all__ = ['CopyPaste', 'copypaste_batch']



def imshow_tensor(tensor):
    plt.figure()
    arr = np.transpose(tensor.cpu().numpy(), (1, 2, 0))
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    plt.imshow(arr)
    

def imshow_1d_tensor(tensor):
    plt.figure()
    arr = tensor.cpu().numpy()
    # arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    plt.imshow(arr+1, cmap="gray")
    

def _aggregate_masks(masks):
    
    aggregated_masks = masks
    return aggregated_masks

def _count_instances(input_tensor):
    unique_class_ids = torch.unique(input_tensor)

    return (len(unique_class_ids) - 1)


def _get_instance_ids(input_tensor, num_instances, filtered_id):
    instance_ids = torch.unique(input_tensor)
    
    npy = instance_ids.cpu().numpy()
    npy = np.delete(npy, np.where(npy == filtered_id))

    rng = default_rng()
    indices = rng.choice(len(npy), size=num_instances, replace=False)

    return npy[indices]



def copypaste_batch(images, masks, configs):
    """
    Randomly pastes objects onto an image.
    """
    batch_idx = 0
    out_images = torch.zeros_like(images)
    out_masks = torch.zeros_like(masks)


    assert images.size(dim=0) == masks.size(dim=0), "Number of images and masks in the batch do not match!"
    batch_size = images.size(dim=0)

    while batch_idx < batch_size:
        [i, j] = np.random.randint(0, high=batch_size, size=2)

        num_instances = _count_instances(masks[i])
        num_copied_instances = random.randint(0, num_instances)
        if configs["max_copied_instances"] is not None:
            num_copied_instances = min(num_copied_instances, configs["max_copied_instances"])

        src_instance_ids = _get_instance_ids(masks[i], num_copied_instances, configs["bg_color"])

        trg_image = images[j]
        trg_mask = masks[j]

        if random.uniform(0, 1) < configs["p"]:
            for src_instance_id in src_instance_ids:
                trg_image, trg_mask = _copypaste_instance(images[i], masks[i], trg_image, trg_mask, src_instance_id, configs)

        out_images[batch_idx] = trg_image
        out_masks[batch_idx] = trg_mask
        
        # fig_name = "copypaste_sample_test" + str(batch_idx)
        # visualize_copypaste_sample(images[i], masks[i], images[j], masks[j], trg_image, trg_mask, fig_name=fig_name)

        batch_idx += 1



    return out_images, out_masks


def visualize_copypaste_sample(src_image, src_mask, trg_image, trg_mask, out_image, out_mask, fig_name=None):
    if fig_name is None:
        fig_name = "copypaste_sample_test"

    dpi = 100
    fig, axarr = plt.subplots(2, 3, figsize=(10, 6), dpi=dpi)


    arr = src_image.cpu().numpy()
    ax = axarr[0, 0]
    ax.set_title("src_image")
    image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    ax.imshow(np.transpose(image, (1, 2, 0)))
    clean_axes(ax)

    arr = src_mask.cpu().numpy()
    ax = axarr[1, 0]
    ax.set_title("src_mask")
    image = arr + 1
    ax.imshow(image, cmap= "gray")
    clean_axes(ax)


    arr = trg_image.cpu().numpy()
    ax = axarr[0, 1]
    ax.set_title("trg_image")
    image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    ax.imshow(np.transpose(image, (1, 2, 0)))
    clean_axes(ax)

    arr = trg_mask.cpu().numpy()
    ax = axarr[1, 1]
    ax.set_title("trg_mask")
    image = arr + 1
    ax.imshow(image, cmap= "gray")
    clean_axes(ax)


    arr = out_image.cpu().numpy()
    ax = axarr[0, 2]
    ax.set_title("out_image")
    image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    ax.imshow(np.transpose(image, (1, 2, 0)))
    clean_axes(ax)

    arr = out_mask.cpu().numpy()
    ax = axarr[1, 2]
    ax.set_title("out_mask")
    image = arr + 1
    ax.imshow(image, cmap= "gray")
    clean_axes(ax)

    plt.suptitle("CopyPaste Augmentation Sample", fontweight="bold")
    fig_out_path = os.path.join(".", "debug_out", "samples")
    
    if not os.path.isdir(fig_out_path):
        os.makedirs(fig_out_path)
    print("sample image saved: ", fig_name)

    plt.savefig(os.path.join(fig_out_path, fig_name + ".png"), dpi=dpi)



def visualize_copypaste_batch(images, masks, out_images, out_masks, num, fig_name=None):
    if fig_name is None:
        fig_name = "copypaste_test"
    start_index = 50
    dpi = 200
    fig, axarr = plt.subplots(4, num, figsize=(14, 8), dpi=dpi)

    for col in range(min(num, len(images))):
        arr = images[start_index + col].cpu().numpy()
        image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        ax = axarr[0, col]
        ax.imshow(np.transpose(image, (1, 2, 0)))
        ax.set_title("images")
        clean_axes(ax)

    for col in range(min(num, len(masks))):
        arr = torch.unsqueeze(masks[start_index + col], dim=0).cpu().numpy()
        image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        ax = axarr[1, col]
        ax.imshow(np.transpose(image, (1, 2, 0)), cmap="gray")
        ax.set_title("masks")
        clean_axes(ax)

    for col in range(min(num, len(out_images))):
        arr = out_images[start_index + col].cpu().numpy()
        # print("out_images: ", (np.max(arr) - np.min(arr)))
        image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        ax = axarr[2, col]
        ax.imshow(np.transpose(image, (1, 2, 0)))
        ax.set_title("out_images")
        clean_axes(ax)

    for col in range(min(num, len(out_masks))):
        arr = torch.unsqueeze(out_masks[start_index + col], dim=0).cpu().numpy()
        # print("out_masks: ", (np.max(arr) - np.min(arr)))
        image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        ax = axarr[3, col]
        ax.imshow(np.transpose(image, (1, 2, 0)), cmap="gray")
        ax.set_title("out_masks")
        clean_axes(ax)

    plt.suptitle("CopyPaste Augmentation Batch", fontweight="bold")
    fig_out_path = os.path.join(".", "debug_out")
    
    if not os.path.isdir(fig_out_path):
        os.makedirs(fig_out_path)

    plt.savefig(os.path.join(fig_out_path, fig_name + ".png"), dpi=dpi)


def clean_axes(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.axis(('tight'))
    ax.set_aspect(aspect=1)



def _decompose_mask(mask, mask_color, background_color):
    mask_npy = mask.numpy()
    unique_vals = np.unique(mask_npy) 
    parsed_mask = torch.zeros([len(unique_vals), mask.size(dim=0), mask.size(dim=1)])

    for i, val in enumerate(unique_vals):
        temp_mask = torch.where(mask == val, mask_color, background_color)
        parsed_mask[i] = temp_mask

    return parsed_mask


def _parse_segmentation_batch(input_dict, mask_color=1, background_color=0):
    for i, aggregated_mask in enumerate(input_dict["aggregated_masks"]):
        input_dict["masks"].append(_decompose_mask(aggregated_mask, mask_color, background_color))

    return input_dict


def save_tensor_to_png(tensor, path, name):
    arr = np.transpose(tensor.cpu().numpy(), (1, 2, 0))
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    plt.imsave(os.path.join(path, name), arr)
    print("Torch tensor saved to png: " + name)


def save_1d_tensor_to_png(tensor, path, name):
    arr = tensor.cpu().numpy()
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    plt.imsave(os.path.join(path, name), arr, cmap="gray")
    print("Torch tensor saved to png: " + name)




class CopyPaste(Algorithm):
    """
    Randomly pastes objects onto an image.

    Args:
        p (float, optional): p. Default: ``0.5``
        convert_to_binary_mask (bool, optional): convert_to_binary_mask. Default: ``True``.
        max_copied_instances (int | None, optional): max_copied_instances. Default: ``None``.
        area_threshold (int, optional): area_threshold. Default: ``100``.
        padding_factor (float, optional): padding_factor. Default: ``0.5``.
        jitter_scale (Tuple[float, float], optional): jitter_scale. Default: ``(0.01, 0.99)``.
        jitter_ratio (Tuple[float, float], optional): jitter_ratio. Default: ``(1.0, 1.0)``.
        p_flip (float, optional): p_flip. Default: ``1.0``.
        bg_color (int, optional): bg_color. Default: ``-1``.
        input_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the input
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair.  The default is 0, which corresponds to any sequence, where the first element
            is the input. Default: ``0``.
        target_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the target
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair. The default is 1, which corresponds to any sequence, where the second element
            is the target. Default: ``1``.

    Example:
        .. testcode::

            from composer.algorithms import CopyPaste
            algorithm = CopyPaste()
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )

    """

    def __init__(
        self,
        p=0.5,
        convert_to_binary_mask=True,
        max_copied_instances=None,
        area_threshold=100,
        padding_factor=0.5,
        jitter_scale=(0.01, 0.99),
        jitter_ratio=(1.0, 1.0),
        p_flip=0.9,
        bg_color=-1,
        input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1,
    ):
        print("------------------------------------")
        print("copypaste constructor is called with p=", p)
        print("------------------------------------")
        self.input_key = input_key
        self.target_key = target_key
        self.configs = {
            "p": p,
            "convert_to_binary_mask": convert_to_binary_mask,
            "max_copied_instances": max_copied_instances,
            "area_threshold": area_threshold,
            "padding_factor": padding_factor,
            "jitter_scale": jitter_scale,
            "jitter_ratio": jitter_ratio,
            "p_flip": p_flip,
            "bg_color": bg_color
        }

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        images = state.batch_get_item(key=self.input_key)
        masks = state.batch_get_item(key=self.target_key)


        batch_num = int(state.timestamp._batch)

        out_images, out_masks = copypaste_batch(images, masks, self.configs)
        
        # visualize_copypaste_batch(images, masks, out_images, out_masks, num=7, fig_name="copypaste_test"+str(batch_num))
        
        state.batch_set_item(key=self.input_key, value=out_images)
        state.batch_set_item(key=self.target_key, value=out_masks)



def _parse_mask_by_id(mask, idx, background_color=-1):
    parsed_mask = torch.where(mask==idx, idx, background_color)

    return parsed_mask



def _copypaste_instance(src_image, src_mask, trg_image, trg_mask, src_instance_id, configs):
    zero_tensor = torch.zeros(1, dtype=src_image.dtype, device=src_image.device)
    bg_color = configs["bg_color"]
    
    src_instance_mask = _parse_mask_by_id(src_mask, src_instance_id, bg_color)
    src_instance = torch.where(src_instance_mask==bg_color, zero_tensor, src_image)


    
    [src_instance, src_instance_mask] = _jitter_instance([src_instance, torch.unsqueeze(src_instance_mask, dim=0)], configs)
    src_instance_mask = torch.squeeze(src_instance_mask)

    # visualize_copypaste_sample(src_instance, src_instance_mask, src_instance, src_instance_mask, src_instance, src_instance_mask, fig_name="debug_" + str(batch_id) + "_" + str(src_instance_id))
    
    trg_image = torch.where(src_instance_mask==bg_color, trg_image, zero_tensor)
    trg_image += src_instance
    trg_mask = torch.where(src_instance_mask==bg_color, trg_mask, src_instance_mask)

    
    return trg_image, trg_mask


def _get_occluded_mask(mask, configs):
    threshold = configs["area_threshold"]

    if _ignore_mask(mask, threshold):
        return torch.ones_like(mask) * configs["bg_color"]
    else:
        return mask

def _get_jitter_transformations(is_mask, padding_size, crop_size, configs):
    fill = 0
    if is_mask == 1:
        fill = configs["bg_color"]

    trns = T.Compose([
        T.Pad(padding=padding_size, fill=fill, padding_mode="constant"),
        T.RandomResizedCrop(size=crop_size, scale=configs["jitter_scale"], ratio=configs["jitter_ratio"], interpolation=T.InterpolationMode.NEAREST), 
        T.RandomHorizontalFlip(p=configs["p_flip"])
        ])

    return trns



def _jitter_instance(arrs, configs):    
    out = []
    jitter_seed = random.randint(0, MAX_TORCH_SEED)

    padding_size = (int(configs["padding_factor"] * arrs[0].size(dim=1)), int(configs["padding_factor"] * arrs[0].size(dim=2)))
    crop_size = (arrs[0].size(dim=1), arrs[0].size(dim=2))
  
    for idx, arr in enumerate(arrs):
        torch.random.manual_seed(jitter_seed)
        random.seed(jitter_seed)
        trns = _get_jitter_transformations(idx, padding_size, crop_size, configs)
        transformed = trns(arr)
        
        out.append(_get_occluded_mask(transformed, configs))
    
    return out


def _ignore_mask(mask, threshold):
    # TODO: here check if mask area is smaller than a threshold, disregard the mask
    # TODO: check if the mask is too big, ignore them (this relates to the abnormal masks in ADE20K)
    print()
    uniques = mask.cpu().numpy()

    print()

    return False

