# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core CopyPaste classes and functions.
TODO: complete the documenetation.

Notes:
- add thresholding for small masks
- mention how you define masks (0 & 1)
-


"""

from __future__ import annotations

import logging
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


def copypaste_batch(input_dict, configs):
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


class CopyPaste(Algorithm):
    """
    Randomly pastes objects onto an image.
    """

    def __init__(
        self,
        input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1,
    ):
        self.input_key = input_key
        self.target_key = target_key

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        input_dict = {
            "images": [],
            "masks": []
        }

        input_dict["images"] = state.batch_get_item(key=self.input_key)
        input_dict["masks"] = state.batch_get_item(key=self.target_key)

        augmented_dict = copypaste_batch(input_dict, configs)  

        state.batch_set_item(key=self.input_key, value=augmented_dict["images"])
        state.batch_set_item(key=self.target_key, value=augmented_dict["masks"])


def _copypaste_instance(input_dict, trg_image, trg_masks, i, src_instance_id, configs):
    # TODO: implement LSJ & SSJ here depending on the hyperparam
    src_masks = input_dict["masks"][i]
    src_instance_mask = src_masks[src_instance_id]
    src_image = input_dict["images"][i]

    if configs["convert_to_binary_mask"]:
        src_instance_mask = torch.where(src_instance_mask==0, 0, 1)

    src_instance = torch.mul(src_image, src_instance_mask)

    src_instance, src_instance_mask = _jitter_instance(src_instance, src_instance_mask, configs)

    trg_image = torch.where(src_instance_mask==0, trg_image, 0)
    trg_image = torch.clamp(trg_image + src_instance, min=0, max=1)

    for idx, trg_mask in enumerate(trg_masks):
        trg_masks[idx] = _get_occluded_mask(src_instance_mask, trg_mask, configs)

    new_trg_mask = torch.unsqueeze(src_instance_mask, dim=0)
    trg_masks = torch.cat((trg_masks, new_trg_mask), dim=0)

    return trg_image, trg_masks


def _get_occluded_mask(src_instance_mask, trg_mask, configs):
    occluded_mask = torch.where(src_instance_mask==1, 0, trg_mask)
    threshold = configs["area_threshold"]

    if _ignore_mask(occluded_mask, threshold):
        return trg_mask
    else:
        return occluded_mask


def _jitter_instance(image, mask, configs):    
    jitter_seed = random.randint(0, MAX_TORCH_SEED)
    torch.random.manual_seed(jitter_seed)
    random.seed(jitter_seed)

    trns = T.Compose([
        T.Pad(padding=(int(configs["padding_factor"] * image.size(dim=1)), int(configs["padding_factor"] * image.size(dim=2))), fill=0, padding_mode="constant"),
        T.RandomResizedCrop(size=(image.size(dim=1), image.size(dim=2)), scale=configs["jitter_scale"], ratio=configs["jitter_ratio"]), 
        T.RandomHorizontalFlip(p=configs["p_flip"])
        ])

    jittered_image = trns(image) 
    torch.random.manual_seed(jitter_seed)
    random.seed(jitter_seed)
    jittered_mask = trns(mask) 

    return jittered_image, jittered_mask


def _ignore_mask(occluded_mask, threshold):
    # TODO: here check if mask area is smaller than a threshold, disregard the mask

    return False