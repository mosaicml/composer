# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ADE20K Semantic segmentation and scene parsing dataset.

Please refer to the `ADE20K dataset <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_ for more details about this
dataset.
"""

import os
from io import BytesIO
from math import ceil
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from composer.datasets.streaming import StreamingDataset

__all__ = ['ADE20k', 'StreamingADE20k']

IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class RandomResizePair(torch.nn.Module):
    """Resize the image and target to ``base_size`` scaled by a randomly sampled value.

    Args:
        min_scale (float): the minimum value the samples can be rescaled.
        max_scale (float): the maximum value the samples can be rescaled.
        base_size (Tuple[int, int]): a specified base size (height x width) to scale to get the resized dimensions.
            When this is None, use the input image size. Default: ``None``.
    """

    def __init__(self, min_scale: float, max_scale: float, base_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.base_size = base_size

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample
        resize_scale = np.random.random_sample() * (self.max_scale - self.min_scale) + self.min_scale
        base_height, base_width = self.base_size if self.base_size else (image.height, image.width)
        resized_dims = (int(base_height * resize_scale), int(base_width * resize_scale))
        resized_image = TF.resize(image, resized_dims, interpolation=TF.InterpolationMode.BILINEAR)  # type: ignore
        resized_target = TF.resize(target, resized_dims, interpolation=TF.InterpolationMode.NEAREST)  # type: ignore
        return resized_image, resized_target


# Based on: https://github.com/open-mmlab/mmsegmentation/blob/aa50358c71fe9c4cccdd2abe42433bdf702e757b/mmseg/datasets/pipelines/transforms.py#L584
class RandomCropPair(torch.nn.Module):
    """Crop the image and target at a randomly sampled position.

    Args:
        crop_size (Tuple[int, int]): the size (height x width) of the crop.
        class_max_percent (float): the maximum percent of the image area a single class should occupy. Default is 1.0.
        num_retry (int): the number of times to resample the crop if ``class_max_percent`` threshold is not reached.
            Default is 1.
    """

    def __init__(self, crop_size: Tuple[int, int], class_max_percent: float = 1.0, num_retry: int = 1):
        super().__init__()
        self.crop_size = crop_size
        self.class_max_percent = class_max_percent
        self.num_retry = num_retry

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample

        # if image size is smaller than crop size, no cropping necessary
        if image.height <= self.crop_size[0] and image.width <= self.crop_size[1]:
            return image, target

        # generate crop
        crop = transforms.RandomCrop.get_params(
            image, output_size=self.crop_size)  # type: ignore - transform typing excludes PIL.Image

        if self.class_max_percent < 1.0:
            for _ in range(self.num_retry):
                # Crop target
                target_crop = TF.crop(target, *crop)  # type: ignore - transform typing excludes PIL.Image

                # count the number of each class represented in cropped target
                labels, counts = np.unique(np.array(target_crop), return_counts=True)
                counts = counts[labels != 0]

                # if the class with the most area is within the class_max_percent threshold, stop retrying
                if len(counts) > 1 and (np.max(counts) / np.sum(counts)) < self.class_max_percent:
                    break

                crop = transforms.RandomCrop.get_params(
                    image, output_size=self.crop_size)  # type: ignore - transform typing excludes PIL.Image

        image = TF.crop(image, *crop)  # type: ignore - transform typing excludes PIL.Image
        target = TF.crop(target, *crop)  # type: ignore - transform typing excludes PIL.Image

        return image, target


class RandomHFlipPair(torch.nn.Module):
    """Flip the image and target horizontally with a specified probability.

    Args:
        probability (float): the probability of flipping the image and target. Default: ``0.5``.
    """

    def __init__(self, probability: float = 0.5):
        super().__init__()
        self.probability = probability

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample
        if np.random.random_sample() > self.probability:
            image = TF.hflip(image)  # type: ignore - transform typing does not include PIL.Image
            target = TF.hflip(target)  # type: ignore - transform typing does not include PIL.Image
        return image, target


class PadToSize(torch.nn.Module):
    """Pad an image to a specified size.

    Args:
        size (Tuple[int, int]): the size (height x width) of the image after padding.
        fill (Union[int, Tuple[int, int, int]]): the value to use for the padded pixels. Default: ``0``.
    """

    def __init__(self, size: Tuple[int, int], fill: Union[int, Tuple[int, int, int]] = 0):
        super().__init__()
        self.size = size
        self.fill = fill

    def forward(self, image: Image.Image):
        padding = max(self.size[0] - image.height, 0), max(self.size[1] - image.width, 0)
        padding = (padding[1] // 2, padding[0] // 2, ceil(padding[1] / 2), ceil(padding[0] / 2))
        image = TF.pad(image, padding, fill=self.fill)  # type: ignore - transform typing does not include PIL.Image
        return image


class PhotometricDistoration(torch.nn.Module):
    """Applies a combination of brightness, contrast, saturation, and hue jitters with random intensity.

    This is a less severe form of PyTorch's ColorJitter used by the mmsegmentation library here:
    https://github.com/open-mmlab/mmsegmentation/blob/aa50358c71fe9c4cccdd2abe42433bdf702e757b/mmseg/datasets/pipelines/transforms.py#L861

    Args:
        brightness (float): max and min to jitter brightness.
        contrast (float): max and min to jitter contrast.
        saturation (float): max and min to jitter saturation.
        hue (float): max and min to jitter hue.
    """

    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, image: Image.Image):
        if np.random.randint(2):
            brightness_factor = np.random.uniform(1 - self.brightness, 1 + self.brightness)
            image = TF.adjust_brightness(
                image, brightness_factor)  # type: ignore - transform typing does not include PIL.Image

        contrast_mode = np.random.randint(2)
        if contrast_mode == 1 and np.random.randint(2):
            contrast_factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            image = TF.adjust_contrast(
                image,  # type: ignore - transform typing does not include PIL.Image
                contrast_factor)

        if np.random.randint(2):
            saturation_factor = np.random.uniform(1 - self.saturation, 1 + self.saturation)
            image = TF.adjust_saturation(
                image, saturation_factor)  # type: ignore - transform typing does not include PIL.Image

        if np.random.randint(2):
            hue_factor = np.random.uniform(-self.hue, self.hue)
            image = TF.adjust_hue(image, hue_factor)  # type: ignore - transform typing does not include PIL.Image

        if contrast_mode == 0 and np.random.randint(2):
            contrast_factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            image = TF.adjust_contrast(
                image,  # type: ignore - transform typing does not include PIL.Image
                contrast_factor)

        return image


class ADE20k(Dataset):
    """PyTorch Dataset for ADE20k.

    Args:
        datadir (str): the path to the ADE20k folder.
        split (str): the dataset split to use, either 'train', 'val', or 'test'. Default: ``'train'``.
        both_transforms (torch.nn.Module): transformations to apply to the image and target simultaneously.
            Default: ``None``.
        image_transforms (torch.nn.Module): transformations to apply to the image only. Default: ``None``.
        target_transforms (torch.nn.Module): transformations to apply to the target only. Default ``None``.
    """

    def __init__(self,
                 datadir: str,
                 split: str = 'train',
                 both_transforms: Optional[torch.nn.Module] = None,
                 image_transforms: Optional[torch.nn.Module] = None,
                 target_transforms: Optional[torch.nn.Module] = None):
        super().__init__()
        self.datadir = datadir
        self.split = split
        self.both_transforms = both_transforms
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

        # Check datadir value
        if self.datadir is None:
            raise ValueError('datadir must be specified')
        elif not os.path.exists(self.datadir):
            raise FileNotFoundError(f'datadir path does not exist: {self.datadir}')

        # Check split value
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f'split must be one of [`train`, `val`, `test`] but is: {self.split}')

        self.image_dir = os.path.join(self.datadir, 'images', self.split)
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f'ADE20k directory structure is not as expected: {self.image_dir} does not exist')

        self.image_files = os.listdir(self.image_dir)

        # Filter for ADE files
        self.image_files = [f for f in self.image_files if f[:3] == 'ADE']

        # Remove grayscale samples
        if self.split == 'train':
            corrupted_samples = ['00003020', '00001701', '00013508', '00008455']
            for sample in corrupted_samples:
                sample_file = f'ADE_train_{sample}.jpg'
                if sample_file in self.image_files:
                    self.image_files.remove(sample_file)

    def __getitem__(self, index):
        # Load image
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path)

        # Load annotation target if using either train or val splits
        if self.split in ['train', 'val']:
            target_path = os.path.join(self.datadir, 'annotations', self.split, image_file.split('.')[0] + '.png')
            target = Image.open(target_path)

            if self.both_transforms:
                image, target = self.both_transforms((image, target))

            if self.target_transforms:
                target = self.target_transforms(target)

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.split in ['train', 'val']:
            return image, target  # type: ignore
        else:
            return image

    def __len__(self):
        return len(self.image_files)


class StreamingADE20k(StreamingDataset):
    """
    Implementation of the ADE20k dataset using StreamingDataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        base_size (int): initial size of the image and target before other augmentations. Default: ``512``.
        min_resize_scale (float): the minimum value the samples can be rescaled. Default: ``0.5``.
        max_resize_scale (float): the maximum value the samples can be rescaled. Default: ``2.0``.
        final_size (int): the final size of the image and target. Default: ``512``.
        batch_size (Optional[int]): Hint the batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def decode_uid(self, data: bytes) -> str:
        return data.decode('utf-8')

    def decode_image(self, data: bytes) -> Image.Image:
        return Image.open(BytesIO(data))

    def decode_annotation(self, data: bytes) -> Image.Image:
        return Image.open(BytesIO(data))

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 base_size: int = 512,
                 min_resize_scale: float = 0.5,
                 max_resize_scale: float = 2.0,
                 final_size: int = 512,
                 batch_size: Optional[int] = None):

        # Validation
        if split not in ['train', 'val']:
            raise ValueError(f"split='{split}' must be one of ['train', 'val'].")
        if base_size <= 0:
            raise ValueError('base_size must be positive.')
        if min_resize_scale <= 0:
            raise ValueError('min_resize_scale must be positive')
        if max_resize_scale <= 0:
            raise ValueError('max_resize_scale must be positive')
        if max_resize_scale < min_resize_scale:
            raise ValueError('max_resize_scale cannot be less than min_resize_scale')
        if final_size <= 0:
            raise ValueError('final_size must be positive')

        # Build StreamingDataset
        decoders = {
            'image': self.decode_image,
            'annotation': self.decode_annotation,
        }
        super().__init__(remote=os.path.join(remote, split),
                         local=os.path.join(local, split),
                         shuffle=shuffle,
                         decoders=decoders,
                         batch_size=batch_size)

        # Define custom transforms
        if split == 'train':
            self.both_transform = torch.nn.Sequential(
                RandomResizePair(min_scale=min_resize_scale,
                                 max_scale=max_resize_scale,
                                 base_size=(base_size, base_size)),
                RandomCropPair(
                    crop_size=(final_size, final_size),
                    class_max_percent=0.75,
                    num_retry=10,
                ),
                RandomHFlipPair(),
            )

            # Photometric distoration values come from mmsegmentation:
            # https://github.com/open-mmlab/mmsegmentation/blob/aa50358c71fe9c4cccdd2abe42433bdf702e757b/mmseg/datasets/pipelines/transforms.py#L861
            r_mean, g_mean, b_mean = IMAGENET_CHANNEL_MEAN
            self.image_transform = torch.nn.Sequential(
                PhotometricDistoration(brightness=32. / 255, contrast=0.5, saturation=0.5, hue=18. / 255),
                PadToSize(size=(final_size, final_size), fill=(int(r_mean), int(g_mean), int(b_mean))))

            self.annotation_transform = PadToSize(size=(final_size, final_size), fill=0)
        else:
            self.both_transform = None
            self.image_transform = transforms.Resize(size=(final_size, final_size),
                                                     interpolation=TF.InterpolationMode.BILINEAR)
            self.annotation_transform = transforms.Resize(size=(final_size, final_size),
                                                          interpolation=TF.InterpolationMode.NEAREST)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        obj = super().__getitem__(idx)
        x = obj['image']
        y = obj['annotation']
        if self.both_transform:
            x, y = self.both_transform((x, y))
        if self.image_transform:
            x = self.image_transform(x)
        if self.annotation_transform:
            y = self.annotation_transform(y)
        return x, y
