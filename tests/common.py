# Copyright 2021 MosaicML. All Rights Reserved.

"""Contains several commonly used objects (models, dataloaders, and batches) that are shared across the test suite."""

from typing import Dict, List, Sequence, Tuple, Union

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

from composer.core.precision import Precision
from composer.models import ComposerClassifier

# syntactic sugar below
# decorators for common parameterizations and marks


def device(*args, precision=False):
    """Decorator for device and optionally precision.

    Input choices are ('cpu', 'gpu'), or if precision=True,
    also accept ('gpu-amp', 'gpu-fp32', and 'cpu-fp32').

    Returns the parameter "device", or if precision=True,
    also returns the parameter "precision".
    """
    # convert cpu-fp32 and gpu-fp32 to cpu, gpu
    if not precision and any(['-' in arg for arg in args]):
        raise ValueError('-fp32 and -amp tags must be removed if precision=False')
    args = [arg.replace('-fp32', '') for arg in args]

    if precision:
        devices = {
            'cpu': pytest.param('cpu', Precision.FP32, id="cpu-fp32"),
            'gpu': pytest.param('gpu', Precision.FP32, id="gpu-fp32", marks=pytest.mark.gpu),
            'gpu-amp': pytest.param('gpu', Precision.AMP, id='gpu-amp', marks=pytest.mark.gpu)
        }
        name = "device,precision"
    else:
        devices = {
            'cpu': pytest.param('cpu', id="cpu"),
            'gpu': pytest.param('gpu', id="gpu", marks=pytest.mark.gpu),
        }
        name = "device"

    parameters = [devices[arg] for arg in args]

    def decorator(test):
        if not parameters:
            return test
        return pytest.mark.parametrize(name, parameters)(test)

    return decorator


def world_size(*args):

    params = {
        1: pytest.param(1),
        2: pytest.param(2, marks=pytest.mark.world_size(2)),
    }
    parameters = [params[arg] for arg in args]

    def decorator(test):
        if not parameters:
            return test
        return pytest.mark.parametrize("world_size", parameters)(test)

    return decorator


class SimpleModel(ComposerClassifier):
    """Small classification model with 5 input features and 2 classes.

    Args:
        num_features (int): number of input features (default: 5)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_features: int = 5, num_classes: int = 2) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        fc1 = torch.nn.Linear(num_features, 5)
        fc2 = torch.nn.Linear(5, num_classes)

        net = torch.nn.Sequential(
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.Softmax(dim=-1),
        )
        super().__init__(module=net)

        # fc1 and fc2 are bound to the model class
        # for access during several surgery tests
        self.fc1 = fc1
        self.fc2 = fc2


class SimpleConvModel(ComposerClassifier):
    """Small convolutional classifer.

    Args:
        num_channels (int): number of input channels (default: 3)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 2) -> None:

        self.num_classes = num_classes
        self.num_channels = num_channels

        conv_args = dict(kernel_size=(3, 3), padding=1)
        conv1 = torch.nn.Conv2d(in_channels=num_channels, out_channels=8, **conv_args)
        conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, **conv_args)
        pool = torch.nn.AdaptiveAvgPool2d(1)
        flatten = torch.nn.Flatten()
        fc1 = torch.nn.Linear(4, 16)
        fc2 = torch.nn.Linear(16, num_classes)

        net = torch.nn.Sequential(
            conv1,
            conv2,
            pool,
            flatten,
            fc1,
            fc2,
        )
        super().__init__(module=net)

        # bind these to class for access during
        # surgery tests
        self.conv1 = conv1
        self.conv2 = conv2


class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: 5)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 100)
    """

    def __init__(self, shape: Sequence[int] = (5,), size: int = 100, num_classes: int = 2):
        self.size = size
        self.x = torch.randn(size, *shape)
        self.y = torch.randint(0, num_classes, size=(size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class RandomImageDataset(VisionDataset):
    """ Image Classification dataset with values drawn from a normal distribution
    Args:
        shape (Sequence[int]): shape of features. Defaults to (32, 32, 3)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 100)
        is_PIL (bool): if true, will emit image in PIL format (default: False)
    """

    def __init__(self, shape: Sequence[int] = (3, 32, 32), size: int = 100, num_classes: int = 2, is_PIL: bool = False):
        self.is_PIL = is_PIL
        if is_PIL:  # PIL expects HWC
            shape = (shape[1], shape[2], shape[0])

        self.size = size
        self.x = torch.randn(size, *shape)
        self.y = torch.randint(0, num_classes, size=(size,))

        super().__init__(root='')

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        x = self.x[index]
        y = self.y[index]

        if self.is_PIL:
            x = x.numpy()
            x = (x - x.min())
            x = (x * (255 / x.max())).astype("uint8")
            x = Image.fromarray(x)

        if self.transform is not None:
            return self.transform(x), y
        else:
            return x, y


def dummy_tensor_batch(batch_size=12) -> torch.Tensor:
    return torch.randn(size=(batch_size, 3, 32, 32))


def dummy_tuple_batch(batch_size=12) -> List[torch.Tensor]:
    # pytorch default collate converts tuples to lists
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L67
    image = torch.randn(size=(batch_size, 3, 32, 32))
    target = torch.randint(size=(batch_size,), high=10)
    return [image, target]


def dummy_tuple_batch_long(batch_size=12) -> List[torch.Tensor]:
    image_1 = torch.randn(size=(batch_size, 3, 32, 32))
    image_2 = torch.randn(size=(batch_size, 3, 32, 32))
    image_3 = torch.randn(size=(batch_size, 3, 32, 32))
    target = torch.randint(size=(batch_size,), high=10)
    return [image_1, image_2, image_3, target]


def dummy_tuple_list_batch(batch_size=12) -> List[Union[List, torch.Tensor]]:
    image_1 = torch.randn(size=(batch_size, 3, 32, 32))
    image_2 = torch.randn(size=(batch_size, 3, 32, 32))
    image_3 = torch.randn(size=(batch_size, 3, 32, 32))
    target = torch.randint(size=(batch_size,), high=10)
    return [[image_1, image_2, image_3], target]


def dummy_dict_batch(batch_size=12) -> Dict[str, torch.Tensor]:
    image = torch.randn(size=(batch_size, 3, 32, 32))
    target = torch.randint(size=(batch_size,), high=10)
    return {'image': image, 'target': target}


def dummy_dict_batch_with_metadata(batch_size=12) -> Dict[str, Union[List, torch.Tensor, str]]:
    # sometimes metadata is included with a batch that isnt taken by the model.
    image = torch.randn(size=(batch_size, 3, 32, 32))
    target = torch.randint(size=(batch_size,), high=10)
    meta = ['hi im a tag' for _ in range(batch_size)]
    index = [[1, 2, 3] for _ in range(batch_size)]
    return {'image': image, 'target': target, 'meta': meta, 'index': index}


def dummy_maskrcnn_batch(batch_size=12,
                         image_height=12,
                         image_width=12,
                         num_classes=80,
                         max_detections=5) -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

    def generate_maskrcnn_sample(num_detections,
                                 image_height=image_height,
                                 image_width=image_width,
                                 num_classes=num_classes):
        """Generates a maskrcnn style sample: (Tensor, Dict[Tensor])."""
        image = torch.randn(size=(3, image_height, image_width)).type(torch.float)
        target = {
            'boxes':
                torch.randint(size=(num_detections, 4), low=0, high=min(image_height, image_width)).type(torch.float),
            'labels':
                torch.randint(size=(num_detections,), low=0, high=num_classes + 1),
            'masks':
                torch.randint(size=(num_detections, image_height, image_width), low=0, high=2).type(torch.uint8)
        }
        return image, target

    return [
        generate_maskrcnn_sample(num_detections=n)
        for n in torch.randint(size=(batch_size,), low=1, high=max_detections + 1)
    ]


def dummy_batches(batch_size=12):
    return [
        dummy_tensor_batch(batch_size=batch_size),
        dummy_tuple_batch(batch_size=batch_size),
        dummy_tuple_batch_long(batch_size=batch_size),
        dummy_tuple_list_batch(batch_size=batch_size),
        dummy_dict_batch(batch_size=batch_size),
        dummy_dict_batch_with_metadata(batch_size=batch_size)
    ]
