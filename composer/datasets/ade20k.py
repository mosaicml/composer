import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
import yahp as hp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from composer.core.types import Batch, Tensor
from composer.datasets.hparams import DataloaderSpec, DatasetHparams


class PreprocessingFn:

    def __init__(self) -> None:
        self.mean: Optional[Tensor] = None
        self.std: Optional[Tensor] = None

    def __call__(self, batch: Batch):
        xs, ys = batch
        assert isinstance(xs, Tensor)
        assert isinstance(ys, Tensor)
        device = xs.device

        if self.mean is None:
            self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255], device=device)
            self.mean = self.mean.view(1, 3, 1, 1)
        if self.std is None:
            self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255], device=device)
            self.std = self.std.view(1, 3, 1, 1)

        xs = xs.float()
        xs = xs.sub_(self.mean).div_(self.std)
        return xs, ys


def fast_collate(batch: List[Tuple[Image.Image, Image.Image]],
                 memory_format: torch.memory_format = torch.contiguous_format):
    imgs = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    tensor_targets = torch.zeros((len(targets), h, w), dtype=torch.long).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)

        nump_array = np.rollaxis(nump_array, 2).copy()
        if nump_array.shape[0] != 3:
            assert nump_array.shape[0] == 1, "unexpected shape"
            nump_array = np.resize(nump_array, (3, h, w))
        assert tuple(tensor.shape)[1:] == nump_array.shape, "shape mistmatch"

        tensor[i] += torch.from_numpy(nump_array)

        target_array = np.asarray(targets[i], dtype=np.int_)
        tensor_targets[i] += target_array  # drop all channels except for the first

    return tensor, (tensor_targets - 1)


class ADE20k(Dataset):

    def __init__(self, datadir: str, split: str, transformation):
        super().__init__()
        self.datadir = datadir
        self.split = split
        self.transformation = transformation

        self.img_files = os.listdir(os.path.join(self.datadir, 'images', self.split))
        # Remove random colo files
        self.img_files = [f for f in self.img_files if f[0] != '.']
        # Remove grayscale image sample
        if self.split == 'train':
            self.img_files.remove('ADE_train_00003020.jpg')
            self.img_files.remove('ADE_train_00001701.jpg')
            self.img_files.remove('ADE_train_00013508.jpg')
            self.img_files.remove('ADE_train_00008455.jpg')

    def __getitem__(self, index):
        # Load image
        #image_name = f'ADE_{self.split}_{index+1:08d}.jpg'
        #image_path = os.path.join(self.datadir, 'images', self.split, image_name)
        image_path = os.path.join(self.datadir, 'images', self.split, self.img_files[index])
        image = Image.open(image_path)

        # Load annotation target if using either train or val splits
        if self.split in ['train', 'val']:
            #target_name = f'ADE_{self.split}_{index+1:08d}.png'
            #target_path = os.path.join(self.datadir, 'annotations', self.split, target_name)
            target_path = os.path.join(self.datadir, 'annotations', self.split,
                                       self.img_files[index].split('.')[0] + '.png')
            target = Image.open(target_path)

        if self.transformation:
            try:
                image, target = self.transformation(image, target, self.split == 'train')
            except:
                print('Error with this image:', image_path)
        return image, target

    def __len__(self):
        return len(self.img_files)


def random_resize(image, target, min_resize_factor, max_resize_factor, factor_step_size):
    resize_factors = np.arange(min_resize_factor, max_resize_factor + factor_step_size, factor_step_size)
    resize_factor = np.random.choice(resize_factors)
    new_dims = [int(image.width * resize_factor), int(image.height * resize_factor)]
    resize_image = TF.resize(image, new_dims, interpolation=TF.InterpolationMode.BILINEAR)
    resize_target = TF.resize(target, new_dims, interpolation=TF.InterpolationMode.NEAREST)
    return resize_image, resize_target


@dataclass
class ADE20kDatasetHparams(DatasetHparams):

    datadir: str = hp.required("Directory containing the dataset")
    split: str = hp.required("Which split of the dataset to use. Either ['train', 'val', 'test']")
    resize_size: int = hp.required("Size of the image after resizing")
    crop_size: int = hp.required("Size of image after cropping")
    drop_last: bool = hp.optional("Whether to drop the last batch from training", default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch", default=True)

    def segmentation_transformation(self, image, target, is_train):
        image = TF.resize(image, [self.resize_size] * 2, TF.InterpolationMode.BILINEAR)
        target = TF.resize(target, [self.resize_size] * 2, TF.InterpolationMode.NEAREST)

        if is_train:
            image, target = random_resize(image, target, 0.5, 2.0, 0.25)

            if image.width < self.crop_size or image.height < self.crop_size:
                margin = self.crop_size - image.width
                image = TF.pad(image, margin // 2, fill=(int(0.485 * 255), int(0.456 * 255), int(0.406 * 255)))
                target = TF.pad(target, margin // 2, fill=0)
            elif image.width > self.crop_size or image.height > self.crop_size:
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
                image = TF.crop(image, i, j, h, w)
                target = TF.crop(target, i, j, h, w)

            if np.random.random() > 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)
        return image, target

    def initialize_object(self) -> DataloaderSpec:

        return DataloaderSpec(dataset=ADE20k(self.datadir, self.split, self.segmentation_transformation),
                              collate_fn=fast_collate,
                              prefetch_fn=PreprocessingFn(),
                              drop_last=self.drop_last,
                              shuffle=self.shuffle)
