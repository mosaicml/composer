# Copyright 2021 MosaicML. All Rights Reserved.

"""ADE20K Semantic segmentation and scene parsing dataset.

Please refer to the `ADE20K dataset <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_ for more details about this
dataset.
"""

import os
from dataclasses import dataclass
from math import ceil
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
import yahp as hp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from composer.core.types import DataSpec
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin, WebDatasetHparams
from composer.datasets.imagenet import IMAGENET_CHANNEL_MEAN, IMAGENET_CHANNEL_STD
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import dist

__all__ = ["ADE20k", "ADE20kDatasetHparams", "ADE20kWebDatasetHparams"]


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


# Based on: https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/pipelines/transforms.py#L584
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
    https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/pipelines/transforms.py#L837

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
            raise ValueError("datadir must be specified")
        elif not os.path.exists(self.datadir):
            raise FileNotFoundError(f"datadir path does not exist: {self.datadir}")

        # Check split value
        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"split must be one of [`train`, `val`, `test`] but is: {self.split}")

        self.image_dir = os.path.join(self.datadir, 'images', self.split)
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"ADE20k directory structure is not as expected: {self.image_dir} does not exist")

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


@dataclass
class ADE20kDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the ADE20k dataset for semantic segmentation from a local disk.

    Args:
        split (str): the dataset split to use either 'train', 'val', or 'test'. Default: ``'train```.
        base_size (int): initial size of the image and target before other augmentations. Default: ``512``.
        min_resize_scale (float): the minimum value the samples can be rescaled. Default: ``0.5``.
        max_resize_scale (float): the maximum value the samples can be rescaled. Default: ``2.0``.
        final_size (int): the final size of the image and target. Default: ``512``.
        ignore_background (bool): if true, ignore the background class when calculating the training loss.
            Default: ``true``.
    """

    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val', 'test']", default='train')
    base_size: int = hp.optional("Initial size of the image and target before other augmentations", default=512)
    min_resize_scale: float = hp.optional("Minimum value that the image and target can be scaled", default=0.5)
    max_resize_scale: float = hp.optional("Maximum value that the image and target can be scaled", default=2.0)
    final_size: int = hp.optional("Final size of the image and target", default=512)
    ignore_background: bool = hp.optional("If true, ignore the background class in training loss", default=True)

    def validate(self):
        if self.datadir is None and not self.use_synthetic:
            raise ValueError("datadir must specify the path to the ADE20k dataset.")

        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"split value {self.split} must be one of ['train', 'val', 'test'].")

        if self.base_size <= 0:
            raise ValueError("base_size cannot be zero or negative.")

        if self.min_resize_scale <= 0:
            raise ValueError("min_resize_scale cannot be zero or negative")

        if self.max_resize_scale < self.min_resize_scale:
            raise ValueError("max_resize_scale cannot be less than min_resize_scale")

    def initialize_object(self, batch_size, dataloader_hparams) -> DataSpec:
        self.validate()

        if self.use_synthetic:
            if self.split == 'train':
                total_dataset_size = 20_206
            elif self.split == 'val':
                total_dataset_size = 2_000
            else:
                total_dataset_size = 3_352

            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, self.final_size, self.final_size],
                label_shape=[self.final_size, self.final_size],
                num_classes=150,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )
            collate_fn = None
            device_transform_fn = None

        else:
            # Define data transformations based on data split
            if self.split == 'train':
                both_transforms = torch.nn.Sequential(
                    RandomResizePair(min_scale=self.min_resize_scale,
                                     max_scale=self.max_resize_scale,
                                     base_size=(self.base_size, self.base_size)),
                    RandomCropPair(
                        crop_size=(self.final_size, self.final_size),
                        class_max_percent=0.75,
                        num_retry=10,
                    ),
                    RandomHFlipPair(),
                )

                # Photometric distoration values come from mmsegmentation:
                # https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/pipelines/transforms.py#L837
                r_mean, g_mean, b_mean = IMAGENET_CHANNEL_MEAN
                image_transforms = torch.nn.Sequential(
                    PhotometricDistoration(brightness=32. / 255, contrast=0.5, saturation=0.5, hue=18. / 255),
                    PadToSize(size=(self.final_size, self.final_size), fill=(int(r_mean), int(g_mean), int(b_mean))))

                target_transforms = PadToSize(size=(self.final_size, self.final_size), fill=0)
            else:
                both_transforms = None
                image_transforms = transforms.Resize(size=(self.final_size, self.final_size),
                                                     interpolation=TF.InterpolationMode.BILINEAR)
                target_transforms = transforms.Resize(size=(self.final_size, self.final_size),
                                                      interpolation=TF.InterpolationMode.NEAREST)
            collate_fn = pil_image_collate
            device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN,
                                                  std=IMAGENET_CHANNEL_STD,
                                                  ignore_background=self.ignore_background)

            # Add check to avoid type ignore below
            if self.datadir is None:
                raise ValueError("datadir must specify the path to the ADE20k dataset.")

            dataset = ADE20k(datadir=self.datadir,
                             split=self.split,
                             both_transforms=both_transforms,
                             image_transforms=image_transforms,
                             target_transforms=target_transforms)
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)
        return DataSpec(dataloader=dataloader_hparams.initialize_object(dataset=dataset,
                                                                        batch_size=batch_size,
                                                                        sampler=sampler,
                                                                        collate_fn=collate_fn,
                                                                        drop_last=self.drop_last),
                        device_transforms=device_transform_fn)


@dataclass
class ADE20kWebDatasetHparams(WebDatasetHparams):
    """Defines an instance of the ADE20k dataset for semantic segmentation from a remote blob store.

    Args:
        remote (str): S3 bucket or root directory where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-ade20k'``
        name (str): Key used to determine where dataset is cached on local filesystem. Default: ``'ade20k'``
        split (str): the dataset split to use either 'train', 'val', or 'test'. Default: ``'train'``.
        base_size (int): initial size of the image and target before other augmentations. Default: ``512``.
        min_resize_scale (float): the minimum value the samples can be rescaled. Default: ``0.5``.
        max_resize_scale (float): the maximum value the samples can be rescaled. Default: ``2.0``.
        final_size (int): the final size of the image and target. Default: ``512``.
        ignore_background (bool): if true, ignore the background class when calculating the training loss.
            Default: ``True``.
    """

    remote: str = hp.optional('WebDataset S3 bucket name', default='s3://mosaicml-internal-dataset-ade20k')
    name: str = hp.optional('WebDataset local cache name', default='ade20k')
    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val', 'test']", default='train')
    base_size: int = hp.optional("Initial size of the image and target before other augmentations", default=512)
    min_resize_scale: float = hp.optional("Minimum value that the image and target can be scaled", default=0.5)
    max_resize_scale: float = hp.optional("Maximum value that the image and target can be scaled", default=2.0)
    final_size: int = hp.optional("Final size of the image and target", default=512)
    ignore_background: bool = hp.optional("If true, ignore the background class in training loss", default=True)

    def validate(self):
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"split value {self.split} must be one of ['train', 'val', 'test'].")

        if self.base_size <= 0:
            raise ValueError("base_size cannot be zero or negative.")

        if self.min_resize_scale <= 0:
            raise ValueError("min_resize_scale cannot be zero or negative")

        if self.max_resize_scale < self.min_resize_scale:
            raise ValueError("max_resize_scale cannot be less than min_resize_scale")

    def initialize_object(self, batch_size, dataloader_hparams) -> DataSpec:
        from composer.datasets.webdataset import load_webdataset

        self.validate()
        # Define data transformations based on data split
        if self.split == 'train':
            both_transforms = torch.nn.Sequential(
                RandomResizePair(min_scale=self.min_resize_scale,
                                 max_scale=self.max_resize_scale,
                                 base_size=(self.base_size, self.base_size)),
                RandomCropPair(
                    crop_size=(self.final_size, self.final_size),
                    class_max_percent=0.75,
                    num_retry=10,
                ),
                RandomHFlipPair(),
            )

            # Photometric distoration values come from mmsegmentation:
            # https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/pipelines/transforms.py#L837
            r_mean, g_mean, b_mean = IMAGENET_CHANNEL_MEAN
            image_transforms = torch.nn.Sequential(
                PhotometricDistoration(brightness=32. / 255, contrast=0.5, saturation=0.5, hue=18. / 255),
                PadToSize(size=(self.final_size, self.final_size), fill=(int(r_mean), int(g_mean), int(b_mean))))

            target_transforms = transforms.Compose([
                PadToSize(size=(self.final_size, self.final_size), fill=0),
                transforms.Grayscale(),
            ])
        else:
            both_transforms = None
            image_transforms = transforms.Resize(size=(self.final_size, self.final_size),
                                                 interpolation=TF.InterpolationMode.BILINEAR)
            target_transforms = transforms.Compose([
                transforms.Resize(size=(self.final_size, self.final_size), interpolation=TF.InterpolationMode.NEAREST),
                transforms.Grayscale(),
            ])

        def map_fn(args):
            x, y = args
            if both_transforms:
                x, y = both_transforms((x, y))
            if image_transforms:
                x = image_transforms(x)
            if target_transforms:
                y = target_transforms(y)
            return x, y

        preprocess = lambda dataset: dataset.decode('pil').to_tuple('scene.jpg', 'annotation.png').map(map_fn)
        dataset = load_webdataset(self.remote, self.name, self.split, self.webdataset_cache_dir,
                                  self.webdataset_cache_verbose, self.shuffle, self.shuffle_buffer, preprocess,
                                  dist.get_world_size(), dataloader_hparams.num_workers, batch_size, self.drop_last)

        collate_fn = pil_image_collate
        device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN,
                                              std=IMAGENET_CHANNEL_STD,
                                              ignore_background=self.ignore_background)

        return DataSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=None,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        ),
                        device_transforms=device_transform_fn)
