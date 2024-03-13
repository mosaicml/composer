# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import torch

from composer.metrics import MAP


def test_map_perfect():
    map = MAP()

    targets = [
        {
            'boxes': torch.tensor([[258.15, 41.29, 606.41, 285.07]]),
            'labels': torch.tensor([4]),
        },  # coco image id 42
        {
            'boxes': torch.tensor([[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]]),
            'labels': torch.tensor([3, 2]),
        },  # coco image id 73
    ]

    # Perfect result
    predictions = [
        {
            'boxes': torch.tensor([[258.15, 41.29, 606.41, 285.07]]),
            'scores': torch.tensor([0.236]),
            'labels': torch.tensor([4]),
        },  # coco image id 42
        {
            'boxes': torch.tensor([[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]]),
            'scores': torch.tensor([0.318, 0.726]),
            'labels': torch.tensor([3, 2]),
        },  # coco image id 73
    ]

    map.update(predictions, targets)
    map_result = map.compute()

    assert map_result['map'] == 1.
    assert map_result['map_50'] == 1.
    assert map_result['map_75'] == 1.


# Completly wrong predictions
def test_map_wront():
    map = MAP()

    targets = [
        {
            'boxes': torch.tensor([[258.15, 41.29, 606.41, 285.07]]),
            'labels': torch.tensor([4]),
        },  # coco image id 42
        {
            'boxes': torch.tensor([[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]]),
            'labels': torch.tensor([3, 2]),
        },  # coco image id 73
    ]

    predictions = [
        {
            'boxes': torch.tensor([[0.0, 3.0, 4.0, 10.0]]),
            'scores': torch.tensor([0.9]),
            'labels': torch.tensor([2]),
        },  # coco image id 42
        {
            'boxes': torch.tensor([[10.0, 15.0, 25.0, 50.0]]),
            'scores': torch.tensor([0.8]),
            'labels': torch.tensor([1]),
        },  # coco image id 73
    ]

    map.update(predictions, targets)
    map_result = map.compute()
    assert map_result['map'] == 0.
    assert map_result['map_50'] == 0.
    assert map_result['map_75'] == 0.


# Imperfect predictions
def test_map_imperfect():
    map = MAP()

    targets = [
        {
            'boxes': torch.tensor([[258.15, 41.29, 606.41, 285.07]]),
            'labels': torch.tensor([4]),
        },  # coco image id 42
        {
            'boxes': torch.tensor([[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]]),
            'labels': torch.tensor([3, 2]),
        },  # coco image id 73
    ]

    # Perfect result
    predictions = [
        {
            'boxes': torch.tensor([[258.15, 41.29, 606.41, 285.07]]),
            'scores': torch.tensor([0.236]),
            'labels': torch.tensor([4]),
        },  # coco image id 42
        {
            'boxes': torch.tensor([[50.0, 22.0, 565.00, 615.0], [12.66, 3.32, 281.26, 275.23]]),
            'scores': torch.tensor([0.318, 0.726]),
            'labels': torch.tensor([3, 1]),
        },  # coco image id 73
    ]

    map.update(predictions, targets)
    map_result = map.compute()
    torch.testing.assert_close(map_result['map'], torch.tensor(0.6667), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(map_result['map_50'], torch.tensor(0.6667), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(map_result['map_75'], torch.tensor(0.6667), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(map_result['map_small'], torch.tensor(-1.))
    torch.testing.assert_close(map_result['map_medium'], torch.tensor(-1.))
    torch.testing.assert_close(map_result['map_large'], torch.tensor(0.6667), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(map_result['mar_1'], torch.tensor(0.6667), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(map_result['mar_10'], torch.tensor(0.6667), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(map_result['mar_100'], torch.tensor(0.6667), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(map_result['mar_small'], torch.tensor(-1.))
    torch.testing.assert_close(map_result['mar_medium'], torch.tensor(-1.))
    torch.testing.assert_close(map_result['mar_large'], torch.tensor(0.6667), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(map_result['map_per_class'], torch.tensor(-1.))
    torch.testing.assert_close(map_result['mar_100_per_class'], torch.tensor(-1.))
