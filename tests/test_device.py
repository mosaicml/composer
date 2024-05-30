# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import pytest
import torch

from composer.devices import DeviceCPU, DeviceGPU
from composer.devices.device import _map_batch
from tests.common import device, world_size


def dummy_tensor_batch() -> torch.Tensor:
    return torch.randn(size=(1, 1, 1, 1))


def dummy_tuple_batch() -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.randn(size=(1, 1, 1, 1))
    target = torch.randint(size=(1,), high=10)
    return image, target


def dummy_tuple_batch_long() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_1 = torch.randn(size=(1, 1, 1, 1))
    image_2 = torch.randn(size=(1, 1, 1, 1))
    image_3 = torch.randn(size=(1, 1, 1, 1))
    target = torch.randint(size=(1,), high=10)
    return image_1, image_2, image_3, target


def dummy_dict_batch() -> dict[str, torch.Tensor]:
    image = torch.randn(size=(1, 1, 1, 1))
    target = torch.randint(size=(1,), high=10)
    return {'image': image, 'target': target}


def dummy_dict_batch_with_metadata(batch_size=1) -> dict[str, Union[list, torch.Tensor, str]]:
    # sometimes metadata is included with a batch that isn't taken by the model.
    image = torch.randn(size=(batch_size, 1, 1, 1))
    target = torch.randint(size=(batch_size,), high=10)
    meta = ['hi im a tag' for _ in range(batch_size)]
    index = [[1, 2, 3] for _ in range(batch_size)]
    return {'image': image, 'target': target, 'meta': meta, 'index': index}


def dummy_maskrcnn_batch() -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:

    def generate_maskrcnn_sample(num_detections, image_height=1, image_width=1, num_classes=1):
        """Generates a maskrcnn style sample: (Tensor, dict[Tensor])."""
        image = torch.randn(size=(3, image_height, image_width)).type(torch.float)
        target = {
            'boxes':
                torch.randint(size=(num_detections, 4), low=0, high=min(image_height, image_width)).type(torch.float),
            'labels':
                torch.randint(size=(num_detections,), low=0, high=num_classes),
            'masks':
                torch.randint(size=(num_detections, image_height, image_width), low=0, high=2).type(torch.uint8),
        }
        return image, target

    def generate_maskrcnn_batch(batch_size, max_detections):
        return [generate_maskrcnn_sample(n) for n in torch.randint(size=(batch_size,), low=1, high=max_detections)]

    return generate_maskrcnn_batch(batch_size=1, max_detections=2)


@device('cpu', 'gpu')
@pytest.mark.parametrize(
    'batch',
    [
        dummy_tensor_batch(),
        dummy_tuple_batch(),
        dummy_tuple_batch_long(),
        dummy_dict_batch(),
        dummy_dict_batch_with_metadata(),
        dummy_maskrcnn_batch(),
    ],
)
def test_to_device(device, batch):
    device_handler = DeviceCPU() if device == 'cpu' else DeviceGPU()

    def assert_device(x):
        if isinstance(x, torch.Tensor):
            assert x.device.type == device_handler._device.type

    new_batch = device_handler.batch_to_device(batch)
    _map_batch(new_batch, assert_device)


@world_size(2)
@device('gpu')
def test_gpu_device_id(device, world_size):
    device_gpu = DeviceGPU(device_id=0)
    assert device_gpu._device.index == 0
