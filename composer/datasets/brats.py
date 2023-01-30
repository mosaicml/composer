# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""BraTS (Brain Tumor Segmentation) dataset.

Please refer to the `Brain Tumor Segmentation (BraTS) challenge <https://www.med.upenn.edu/cbica/brats2021/>`_ for more
details about this dataset.
"""

import glob
import os
import random

import numpy as np
import torch
import torch.utils.data
import torchvision

from composer.utils import MissingConditionalImportError, dist

PATCH_SIZE = [1, 192, 160]

__all__ = ['PytTrain', 'PytVal']


def build_brats_dataloader(datadir: str,
                           global_batch_size: int,
                           oversampling: float = 0.33,
                           is_train: bool = True,
                           drop_last: bool = True,
                           shuffle: bool = True,
                           **dataloader_kwargs):
    """Builds a BRaTS dataloader

    Args:
        global_batch_size (int): Global batch size.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()
    x_train, y_train, x_val, y_val = get_data_split(datadir)
    dataset = PytTrain(x_train, y_train, oversampling) if is_train else PytVal(x_val, y_val)
    collate_fn = None if is_train else _my_collate
    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)

    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       sampler=sampler,
                                       drop_last=drop_last,
                                       collate_fn=collate_fn,
                                       **dataloader_kwargs)


def _my_collate(batch):
    """Custom collate function to handle images with different depths."""
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]

    return [torch.Tensor(data), torch.Tensor(target)]


def _coin_flip(prob):
    return random.random() < prob


def _random_augmentation(probability, augmented, original):
    condition = _coin_flip(probability)
    neg_condition = condition ^ True
    return condition * augmented + neg_condition * original


class Crop(object):

    def __call__(self, data, oversampling):
        img, lbl = data['image'], data['label']

        def randrange(max_range):
            return 0 if max_range == 0 else random.randrange(max_range)

        def get_cords(cord, idx):
            return cord[idx], cord[idx] + PATCH_SIZE[idx]

        def _rand_crop(image, label):
            ranges = [s - p for s, p in zip(image.shape[1:], PATCH_SIZE)]

            cord = [randrange(x) for x in ranges]
            low_x, high_x = get_cords(cord, 0)
            low_y, high_y = get_cords(cord, 1)
            image = image[:, low_x:high_x, low_y:high_y]
            label = label[:, low_x:high_x, low_y:high_y]
            return image, label, [low_x, high_x, low_y, high_y]

        def rand_foreg_cropd(image, label):

            import scipy.ndimage
            cl = np.random.choice(np.unique(label[label > 0]))
            foreg_slices = scipy.ndimage.find_objects(scipy.ndimage.measurements.label(label == cl)[0])
            foreg_slices = [x for x in foreg_slices if x is not None]
            slice_volumes = [np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices]
            slice_idx = np.argsort(slice_volumes)[-2:]
            foreg_slices = [foreg_slices[i] for i in slice_idx]
            if not foreg_slices:
                return _rand_crop(image, label)
            foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
            low_x, high_x = adjust(foreg_slice, PATCH_SIZE, label, 1)
            low_y, high_y = adjust(foreg_slice, PATCH_SIZE, label, 2)
            image = image[:, low_x:high_x, low_y:high_y]
            label = label[:, low_x:high_x, low_y:high_y]
            return image, label, [low_x, high_x, low_y, high_y]

        def adjust(foreg_slice, patch_size, label, idx):

            diff = patch_size[idx - 1] - (foreg_slice[idx].stop - foreg_slice[idx].start)
            sign = -1 if diff < 0 else 1
            diff = abs(diff)
            ladj = randrange(diff)
            hadj = diff - ladj
            low = max(0, foreg_slice[idx].start - sign * ladj)
            high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
            diff = patch_size[idx - 1] - (high - low)
            if diff > 0 and low == 0:
                high += diff
            elif diff > 0:
                low -= diff
            return low, high

        if random.random() < oversampling:
            img, lbl, _ = rand_foreg_cropd(img, lbl)
        else:
            img, lbl, _ = _rand_crop(img, lbl)

        return {'image': img, 'label': lbl}


class Noise(object):

    def __call__(self, data, oversampling):
        img, lbl = data['image'], data['label']
        std = np.random.uniform(0.0, oversampling)
        noise = np.random.normal(0, scale=std, size=img.shape)
        img_noised = img + noise
        img = _random_augmentation(0.15, img_noised, img)

        return {'image': img, 'label': lbl}


class Blur(object):

    def __call__(self, data):
        img, lbl = data['image'], data['label']

        transf = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))
        img_blured = transf(torch.Tensor(img)).numpy()
        img = _random_augmentation(0.15, img_blured, img)

        return {'image': img, 'label': lbl}


class Brightness(object):

    def __call__(self, data):
        img, lbl = data['image'], data['label']
        brightness_scale = _random_augmentation(0.15, np.random.uniform(0.7, 1.3), 1.0)
        img = img * brightness_scale

        return {'image': img, 'label': lbl}


class Contrast(object):

    def __call__(self, data):
        img, lbl = data['image'], data['label']
        min_, max_ = np.min(img), np.max(img)
        scale = _random_augmentation(0.15, np.random.uniform(0.65, 1.5), 1.0)

        img = torch.clamp(torch.Tensor(img * scale), min_, max_).numpy()
        return {'image': img, 'label': lbl}


class Flips(object):

    def __call__(self, data):
        img, lbl = data['image'], data['label']
        axes = [1, 2]
        prob = 1 / len(axes)

        for axis in axes:
            if random.random() < prob:
                img = np.flip(img, axis=axis).copy()
                lbl = np.flip(lbl, axis=axis).copy()

        return {'image': img, 'label': lbl}


class Transpose(object):

    def __call__(self, data):
        img, lbl = data['image'], data['label']
        img, lbl = img.transpose((1, 0, 2, 3)), lbl.transpose((1, 0, 2, 3))

        return {'image': img, 'label': lbl}


class PytTrain(torch.utils.data.Dataset):

    def __init__(self, images, labels, oversampling, transform=None):
        self.images, self.labels = images, labels
        self.oversampling = oversampling
        self.transform = transform
        self.rand_crop = Crop()
        self.transpose = Transpose()
        self.contrast = Contrast()
        self.noise = Noise()
        self.blur = Blur()
        self.flips = Flips()
        self.bright = Brightness()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = {'image': np.load(self.images[idx]), 'label': np.load(self.labels[idx])}
        data = self.rand_crop(data, self.oversampling)
        data = self.flips(data)
        data = self.noise(data, self.oversampling)
        data = self.blur(data)
        data = self.bright(data)
        data = self.contrast(data)
        data = self.transpose(data)

        return data['image'], data['label']


class PytVal(torch.utils.data.Dataset):

    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = {'image': np.load(self.images[idx]), 'label': np.load(self.labels[idx])}
        return data['image'], data['label']


def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f'Found no data at {path}'
    return data


def get_split(data, idx):
    return list(np.array(data)[idx])


def get_data_split(path: str):
    try:
        from sklearn.model_selection import KFold
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='unet',
                                            conda_channel='conda-forge',
                                            conda_package='scikit-learn') from e
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    imgs = load_data(path, '*_x.npy')
    lbls = load_data(path, '*_y.npy')
    assert len(imgs) == len(lbls), f'Found {len(imgs)} volumes but {len(lbls)} corresponding masks'
    train_imgs, train_lbls, val_imgs, val_lbls = [], [], [], []

    train_idx, val_idx = list(kfold.split(imgs))[0]
    train_imgs = get_split(imgs, train_idx)
    train_lbls = get_split(lbls, train_idx)
    val_imgs = get_split(imgs, val_idx)
    val_lbls = get_split(lbls, val_idx)

    return train_imgs, train_lbls, val_imgs, val_lbls
