# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core CopyPaste classes and functions.
TODO: complete the documenetation.

Notes:
- add thresholding for small masks
- mention how you define masks (0 & 1)
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
    


def copypaste_batch_backup(input_dict, configs):
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
        if configs["max_copied_instances"] is not None:
            num_copied_instances = min(num_copied_instances, configs["max_copied_instances"])

        rng = default_rng()
        src_instance_ids = rng.choice(num_instances, size=num_copied_instances, replace=False)

        trg_image = input_dict["images"][j]
        trg_masks = input_dict["masks"][j]

        if random.uniform(0, 1) < configs["p"]:
            for idx in range(num_copied_instances):
                trg_image, trg_masks = _copypaste_instance(input_dict, trg_image, trg_masks, i, src_instance_ids[idx], configs)

        output_dict["images"].append(trg_image)
        output_dict["masks"].append(trg_masks)

    return output_dict


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
    counter = 0
    out_images = torch.zeros_like(images)
    out_masks = torch.zeros_like(masks)


    assert images.size(dim=0) == masks.size(dim=0), "Number of images and masks in the batch do not match!"
    batch_size = images.size(dim=0)

    while counter < batch_size:
        [i, j] = np.random.randint(0, high=batch_size, size=2)
        i=1
        j=0
        # num_instances = input_dict["masks"][i].shape[0]

        num_instances = _count_instances(masks[i])
        num_copied_instances = random.randint(0, num_instances)
        if configs["max_copied_instances"] is not None:
            num_copied_instances = min(num_copied_instances, configs["max_copied_instances"])


        src_instance_ids = _get_instance_ids(masks[i], num_copied_instances, configs["bg_color"])


        trg_image = images[j]
        trg_mask = masks[j]

        # imshow_1d_tensor(masks[i])
        # imshow_tensor(images[i])
        # imshow_1d_tensor(trg_mask)
        # imshow_tensor(trg_image)
        # print("source: ", torch.unique(masks[i]))
        # print("target: ", torch.unique(trg_mask))
        if random.uniform(0, 1) < configs["p"]:
            for src_instance_id in src_instance_ids:


                src_instance_id = 1
                trg_image, trg_mask = _copypaste_instance(images[i], masks[i], trg_image, trg_mask, src_instance_id, configs)

        # imshow_1d_tensor(trg_mask)
        # imshow_tensor(trg_image)
        # print("after: ", torch.unique(trg_mask))
        # print()


        # aggregated_trg_masks = _aggregate_masks(trg_masks)
        # output_dict["images"].append(trg_image)
        # output_dict["aggregated_masks"].append(aggregated_trg_masks)
    visualize_copypaste_batch()

    return out_images, out_masks


def visualize_copypaste_batch(output_dict, input_dict, batch_size, i, j, fig_name=None):
    if fig_name is None:
        fig_name = "SRC-"+input_dict["sample_names"][i]+"-TRG-"+input_dict["sample_names"][j]

    dpi = 150

    fig, axarr = plt.subplots(2, batch_size, figsize=(18, 7), dpi=dpi)

    for col, image in enumerate(output_dict["images"]):

        ax = axarr[1, col]
        ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        clean_axes(ax)

    for col, image in enumerate(input_dict["images"]):
        ax = axarr[0, col]
        ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        clean_axes(ax)
        

    plt.suptitle("CopyPaste Augmentation Batch", fontweight="bold")

    fig_out_path = os.path.join(".", "forks", "composer", "composer", "algorithms", "copypaste", "files", "out", "no_jittering", fig_name)
    plt.savefig(fig_out_path + ".png", dpi=dpi)


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
        p (float, optional): p. Default: ``1.0``
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
        p=1.0,
        convert_to_binary_mask=True,
        max_copied_instances=None,
        area_threshold=100,
        padding_factor=0.5,
        jitter_scale=(0.01, 0.99),
        jitter_ratio=(1.0, 1.0),
        p_flip=1.0,
        input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1,
    ):
        print("------------------------------------")
        print("copypaste constructor is called")
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
            "p_flip": p_flip
        }

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        print()
        print("------------------------------------")
        print("apply is called")
        print("------------------------------------")

        input_dict = {
            "images": [],
            "aggregated_masks": []
        }

        input_dict["images"] = state.batch_get_item(key=self.input_key)
        input_dict["aggregated_masks"] = state.batch_get_item(key=self.target_key)

        test_image = input_dict["images"][2]
        test_mask = input_dict["aggregated_masks"][2]

        # print(torch.min(test_image), " - ", torch.max(test_image))
        # print(torch.min(test_mask), " - ", torch.max(test_mask))
        
        path = os.path.join(".", "debug_out")
        if not os.path.isdir(path):
            os.makedirs(path)
        
        save_tensor_to_png(test_image, path, "image2.png")
        save_1d_tensor_to_png(test_mask, path, "mask2.png")



        input_dict = _parse_segmentation_batch(input_dict)
        augmented_dict = copypaste_batch(input_dict, self.configs)


        # here consolidate["masks"] to one flatten mask. you need to preserve mask values for this.
        # state.batch_set_item(key=self.input_key, value=augmented_dict["images"])
        # state.batch_set_item(key=self.target_key, value=augmented_dict["masks"])




        ## bypassing CopyPaste to make it run
        state.batch_set_item(key=self.input_key, value=state.batch_get_item(key=self.input_key))
        state.batch_set_item(key=self.target_key, value=state.batch_get_item(key=self.target_key))




class CopyPaste_backup(Algorithm):
    """
    Randomly pastes objects onto an image.
    """

    def __init__(
        self,
        p=1.0,
        convert_to_binary_mask=True,
        max_copied_instances=None,
        area_threshold=100,
        padding_factor=0.5,
        jitter_scale=(0.01, 0.99),
        jitter_ratio=(1.0, 1.0),
        p_flip=1.0,
        bg_color=-1,
        input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1,
    ):
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
        input_dict = {
            "images": [],
            "masks": []
        }

        input_dict["images"] = state.batch_get_item(key=self.input_key)
        input_dict["masks"] = state.batch_get_item(key=self.target_key)

        augmented_dict = copypaste_batch(input_dict, self.configs)  

        state.batch_set_item(key=self.input_key, value=augmented_dict["images"])
        state.batch_set_item(key=self.target_key, value=augmented_dict["masks"])


def _parse_mask_by_id(mask, idx, background_color=-1):
    parsed_mask = torch.where(mask == idx, idx, background_color)

    return parsed_mask



def _copypaste_instance(src_image, src_mask, trg_image, trg_mask, src_instance_id, configs):
    # TODO: move to configs
    bg_color = configs["bg_color"]
    # src_instance_mask = src_mask[src_instance_id]
    src_instance_mask = _parse_mask_by_id(src_mask, src_instance_id, bg_color)
    # if configs["convert_to_binary_mask"]:
    #     src_instance_mask = torch.where(src_instance_mask==0, 0, 1)

    # src_instance = torch.mul(src_image, src_instance_mask)

    src_instance = torch.where(src_instance_mask==bg_color, 0, src_image)

    # imshow_tensor(src_instance)
    # imshow_1d_tensor(src_instance_mask)
    # print("debug1:", torch.unique(src_instance_mask))
    [src_instance, src_instance_mask] = _jitter_instance([src_instance, torch.unsqueeze(src_instance_mask, dim=0)], configs)
    
    # print("debug2:", torch.unique(src_instance_mask))
    src_instance_mask = torch.squeeze(src_instance_mask)
    # print("debug3:", torch.unique(src_instance_mask))
    # imshow_tensor(src_instance)
    # imshow_1d_tensor(src_instance_mask)

    trg_image = torch.where(src_instance_mask==bg_color, trg_image, 0)
    trg_image += src_instance
    trg_mask = torch.where(src_instance_mask==bg_color, trg_mask, src_instance_mask)

    # for idx, trg_mask in enumerate(trg_masks):
    #     trg_masks[idx] = _get_occluded_mask(src_instance_mask, trg_mask, configs)

    # new_trg_mask = torch.unsqueeze(src_instance_mask, dim=0)
    # trg_masks = torch.cat((trg_masks, new_trg_mask), dim=0)

    return trg_image, trg_mask


def _get_occluded_mask(src_instance_mask, trg_mask, configs):
    occluded_mask = torch.where(src_instance_mask==1, 0, trg_mask)
    threshold = configs["area_threshold"]

    if _ignore_mask(occluded_mask, threshold):
        return trg_mask
    else:
        return occluded_mask


def _jitter_instance(arrs, configs):    
    out = []
    jitter_seed = random.randint(0, MAX_TORCH_SEED)
    
    padding_size = (int(configs["padding_factor"] * arrs[0].size(dim=1)), int(configs["padding_factor"] * arrs[0].size(dim=2)))
    crop_size = (arrs[0].size(dim=1), arrs[0].size(dim=2))

    trns = T.Compose([
        T.Pad(padding=padding_size, fill=0, padding_mode="constant"),
        T.RandomResizedCrop(size=crop_size, scale=configs["jitter_scale"], ratio=configs["jitter_ratio"], interpolation=T.InterpolationMode.NEAREST), 
        T.RandomHorizontalFlip(p=configs["p_flip"])
        ])

    for arr in arrs:
        torch.random.manual_seed(jitter_seed)
        random.seed(jitter_seed)
        transformed = trns(arr)
        out.append(transformed)

    return out


def _ignore_mask(occluded_mask, threshold):
    # TODO: here check if mask area is smaller than a threshold, disregard the mask
    # TODO: check if the mask is too big, ignore them (this relates to the abnormal masks in ADE20K)
    return False

