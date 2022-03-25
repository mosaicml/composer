from typing import Dict, List, Tuple

import pytest
import torch

from composer.trainer.devices import DeviceCPU, DeviceGPU
from composer.trainer.devices.device import _map_collections
from tests.common import device


def dummy_tensor_batch() -> torch.Tensor:
    return torch.randn(size=(12, 3, 32, 32))


def dummy_tuple_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    image = torch.randn(size=(12, 3, 32, 32))
    target = torch.randint(size=(12,), high=10)
    return image, target


def dummy_tuple_batch_long() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_1 = torch.randn(size=(12, 3, 32, 32))
    image_2 = torch.randn(size=(12, 3, 32, 32))
    image_3 = torch.randn(size=(12, 3, 32, 32))
    target = torch.randint(size=(12,), high=10)
    return image_1, image_2, image_3, target


def dummy_dict_batch() -> Dict[str, torch.Tensor]:
    image = torch.randn(size=(12, 3, 32, 32))
    target = torch.randint(size=(12,), high=10)
    return {'image': image, 'target': target}


def dummy_maskrcnn_batch() -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

    def generate_maskrcnn_sample(num_detections, image_height=12, image_width=12, num_classes=80):
        """Generates a maskrcnn style sample: (Tensor, Dict[Tensor])."""
        image = torch.randn(size=(3, image_height, image_width)).type(torch.float)
        target = {
            'boxes':
                torch.randint(size=(num_detections, 4), low=0, high=min(image_height, image_width)).type(torch.float),
            'labels':
                torch.randint(size=(num_detections,), low=0, high=num_classes),
            'masks':
                torch.randint(size=(num_detections, image_height, image_width), low=0, high=2).type(torch.uint8)
        }
        return image, target

    def generate_maskrcnn_batch(batch_size, max_detections):
        return [generate_maskrcnn_sample(n) for n in torch.randint(size=(batch_size,), low=1, high=max_detections)]

    return generate_maskrcnn_batch(batch_size=5, max_detections=5)


@device('cpu', 'gpu')
@pytest.mark.parametrize(
    "batch",
    [dummy_tensor_batch(),
     dummy_tuple_batch(),
     dummy_tuple_batch_long(),
     dummy_dict_batch(),
     dummy_maskrcnn_batch()])
def test_to_device(device, batch):
    device_handler = DeviceCPU() if device == 'cpu' else DeviceGPU()

    def assert_device(x):
        if isinstance(x, torch.Tensor):
            assert x.device.type == device_handler._device

    new_batch = device_handler.batch_to_device(batch)
    _map_collections(new_batch, assert_device)
