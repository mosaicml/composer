from typing import List, Tuple, Dict

import pytest
import torch
from composer.trainer.devices import DeviceCPU, DeviceGPU


def dummy_tensor_batch() -> torch.Tensor:
    return torch.randn(size=(12, 3, 32, 32))


def dummy_tuple_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    image = torch.randn(size=(12, 3, 32, 32))
    target = torch.randint(size=(12,), high=10)
    return image, target


def dummy_dict_batch() -> Dict[str, torch.Tensor]:
    image = torch.randn(size=(12, 3, 32, 32))
    target = torch.randint(size=(12,), high=10)
    return {'image': image, 'target': target}


def dummy_maskrcnn_batch() -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    def generate_maskrcnn_sample(num_detections, image_height=12, image_width=12, num_classes=80):
        """Generates a maskrcnn style sample (Tensor, Dict[Tensor])."""
        image = torch.randn(size=(3, image_height, image_width)).type(torch.float)
        target = {'boxes': torch.randint(size=(num_detections, 4), low=0, high=min(image_height, image_width)).type(torch.float),
                  'labels': torch.randint(size=(num_detections,), low=0, high=num_classes),
                  'masks': torch.randint(size=(num_detections, image_height, image_width), low=0, high=2).type(torch.uint8)}
        return image, target

    def generate_maskrcnn_batch(batch_size, max_detections):
        return [generate_maskrcnn_sample(n) for n in torch.randint(size=(batch_size,), low=1, high=max_detections)]

    return generate_maskrcnn_batch(batch_size=5, max_detections=5)


@pytest.mark.parametrize("batch", [dummy_tensor_batch(),
                                   dummy_tuple_batch(),
                                   dummy_dict_batch(),
                                   dummy_maskrcnn_batch()])
def test_to_device_cpu(batch):
    device = DeviceCPU()
    new_batch = device.batch_to_device(batch)
    # test something about new batch?


@pytest.mark.parametrize("batch", [dummy_tensor_batch(),
                                   dummy_tuple_batch(),
                                   dummy_dict_batch(),
                                   dummy_maskrcnn_batch()])
@pytest.mark.gpu()
def test_to_device_gpu(batch):
    device = DeviceGPU()
    new_batch = device.batch_to_device(batch)
    # test something about new batch?
