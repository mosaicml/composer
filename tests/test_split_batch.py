# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Mapping, Union

import pytest
import torch

from composer.core.data_spec import _split_list, _split_tensor, default_split_batch


def dummy_tensor_batch(batch_size=12) -> torch.Tensor:
    return torch.randn(size=(batch_size, 3, 32, 32))


def dummy_list_str(batch_size=12) -> list[str]:
    return [str(x) for x in range(batch_size)]


def dummy_tuple_batch(batch_size=12) -> list[torch.Tensor]:
    # pytorch default collate converts tuples to lists
    # https://github.com/pytorch/pytorch/blob/e451259a609acdcd83105177ddba73fc41cfa9b4/torch/utils/data/_utils/collate.py#L67
    image = torch.randn(size=(batch_size, 3, 32, 32))
    target = torch.randint(size=(batch_size,), high=10)
    return [image, target]


def dummy_tuple_batch_long(batch_size=12) -> list[torch.Tensor]:
    image_1 = torch.randn(size=(batch_size, 3, 32, 32))
    image_2 = torch.randn(size=(batch_size, 3, 32, 32))
    image_3 = torch.randn(size=(batch_size, 3, 32, 32))
    target = torch.randint(size=(batch_size,), high=10)
    return [image_1, image_2, image_3, target]


def dummy_dict_batch(batch_size=12) -> dict[str, torch.Tensor]:
    image = torch.randn(size=(batch_size, 3, 32, 32))
    target = torch.randint(size=(batch_size,), high=10)
    return {'image': image, 'target': target}


def dummy_dict_batch_with_metadata(batch_size=12) -> dict[str, Union[list, torch.Tensor, str]]:
    # sometimes metadata is included with a batch that isn't taken by the model.
    image = torch.randn(size=(batch_size, 3, 32, 32))
    target = torch.randint(size=(batch_size,), high=10)
    meta = ['hi im a tag' for _ in range(batch_size)]
    index = [[1, 2, 3] for _ in range(batch_size)]
    return {'image': image, 'target': target, 'meta': meta, 'index': index}


def dummy_dict_batch_with_common_metadata(batch_size=12) -> dict[str, Union[list, torch.Tensor, str]]:
    # sometimes metadata is included with a batch that isn't taken by the model.
    image = torch.randn(size=(batch_size, 3, 32, 32))
    target = torch.randint(size=(batch_size,), high=10)
    meta = 'this is a string'
    index = [[1, 2, 3] for _ in range(batch_size)]
    return {'image': image, 'target': target, 'meta': meta, 'index': index}


def dummy_maskrcnn_batch(
    batch_size=12,
    image_height=12,
    image_width=12,
    num_classes=80,
    max_detections=5,
) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:

    def generate_maskrcnn_sample(
        num_detections,
        image_height=image_height,
        image_width=image_width,
        num_classes=num_classes,
    ):
        """Generates a maskrcnn style sample: (Tensor, dict[Tensor])."""
        image = torch.randn(size=(3, image_height, image_width)).type(torch.float)
        target = {
            'boxes':
                torch.randint(size=(num_detections, 4), low=0, high=min(image_height, image_width)).type(torch.float),
            'labels':
                torch.randint(size=(num_detections,), low=0, high=num_classes + 1),
            'masks':
                torch.randint(size=(num_detections, image_height, image_width), low=0, high=2).type(torch.uint8),
        }
        return image, target

    return [
        generate_maskrcnn_sample(num_detections=n)
        for n in torch.randint(size=(batch_size,), low=1, high=max_detections + 1)
    ]


def dummy_batches(batch_size=12):
    return [
        dummy_tensor_batch(batch_size=batch_size),
        dummy_list_str(batch_size=batch_size),
        dummy_tuple_batch(batch_size=batch_size),
        dummy_tuple_batch_long(batch_size=batch_size),
        dummy_dict_batch(batch_size=batch_size),
        dummy_dict_batch_with_metadata(batch_size=batch_size),
        dummy_dict_batch_with_common_metadata(batch_size=batch_size),
    ]


@pytest.mark.parametrize('batch', dummy_batches(12))
def test_split_without_error(batch):
    microbatches = default_split_batch(batch, microbatch_size=3)
    assert len(microbatches) == 4


@pytest.mark.parametrize('batch', [dummy_tensor_batch(i) for i in [12, 13, 14, 15]])
def test_tensor_vs_list_chunking(batch):
    tensor_microbatches = _split_tensor(batch, microbatch_size=4)
    list_microbatches = _split_list(list(batch), microbatch_size=4)

    assert len(tensor_microbatches) == len(list_microbatches)
    assert all(torch.equal(t1, torch.stack(t2, dim=0)) for t1, t2 in zip(tensor_microbatches, list_microbatches))


@pytest.mark.parametrize('batch', [dummy_tuple_batch(12)])
def test_split_tuple(batch):
    microbatches = default_split_batch(batch, microbatch_size=4)
    # should be 3 microbatches of size 4 tensors pairs
    # should split into [(x, y), (x, y), (x, y)]
    assert len(microbatches[0]) == 2


@pytest.mark.parametrize('batch', [dummy_tuple_batch_long(12)])
def test_split_tuple_long(batch):
    microbatches = default_split_batch(batch, microbatch_size=4)
    assert len(microbatches[0]) == 4


@pytest.mark.parametrize('batch', dummy_batches(6))
def test_batch_sizes(batch):
    microbatches = default_split_batch(batch, microbatch_size=2)
    # should split into [len(2), len(2), len(1)]
    assert len(microbatches) == 3
    for microbatch in microbatches:
        if isinstance(microbatch, Mapping):
            assert len(microbatch['image']) == 2
            assert len(microbatch['target']) == 2
        if isinstance(microbatch, tuple):
            assert len(microbatch[0]) == 2
        if isinstance(microbatch, list):
            assert len(microbatch) == 2


@pytest.mark.parametrize('batch', dummy_batches(5))
def test_odd_batch_sizes(batch):
    microbatches = default_split_batch(batch, microbatch_size=2)
    # should split into [len(2), len(2), len(1)]
    assert len(microbatches) == 3
    last_microbatch = microbatches[-1]
    if isinstance(last_microbatch, Mapping):
        assert len(last_microbatch['image']) == 1
        assert len(last_microbatch['target']) == 1
    if isinstance(last_microbatch, tuple):
        assert len(last_microbatch[0]) == 1
    if isinstance(last_microbatch, list):
        assert len(last_microbatch) == 1


@pytest.mark.parametrize('batch', dummy_batches(2))
def test_microbatch_size_greater_than_batch_size(batch):
    with pytest.warns(UserWarning):
        microbatches = default_split_batch(batch, microbatch_size=3)
        assert len(microbatches) == 1


@pytest.mark.parametrize('batch', [dummy_maskrcnn_batch(12)])
def test_microbatch_size_split_maskrcnn(batch):
    microbatches = _split_list(batch, microbatch_size=4)
    assert len(microbatches) == 3


@pytest.mark.parametrize('batch', [dummy_dict_batch_with_common_metadata(12)])
def test_primitive_broadcast(batch):
    microbatches = default_split_batch(batch, microbatch_size=3)
    assert len(microbatches) == 4
    for mb in microbatches:
        assert mb['meta'] == 'this is a string'
