# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import datetime
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from composer import Time
from composer.core.time import TimeUnit


def deep_compare(item1: Any, item2: Any, atol: float = 0.0, rtol: float = 0.0):
    """Compare two items recursively. Supports dicts, lists, tuples, tensors, numpy arrays, Composer Time objects, and callables.

    Args:
        item1 (Any): The first item
        item2 (Any): The second item
        atol (bool): Atol tolerance for torch tensors and numpy arrays (default: 0.0)
        rtol (float): Rtol tolerance for torch tensors and numpy arrays (default: 0.0)
    """
    return _check_item(item1, item2, path='', atol=atol, rtol=rtol)


def _check_item(item1: Any, item2: Any, path: str, rtol: float = 0.0, atol: float = 0.0):
    if item1 is None:
        assert item2 is None, f'{path} differs: {item1} != {item2}'
        return
    if isinstance(item1, (str, float, int, bool, Time, datetime.timedelta, TimeUnit)):
        assert type(item1) == type(item2)
        assert item1 == item2, f'{path} differs: {item1} != {item2}'
        return
    if isinstance(item1, torch.Tensor):
        assert isinstance(item2, torch.Tensor)
        assert item1.allclose(item2, rtol=rtol, atol=atol), f'{path} differs'
        return
    if isinstance(item1, np.ndarray):
        assert isinstance(item2, np.ndarray)
        assert np.allclose(item1, item2, rtol=0.1, atol=0.1), f'{path} differs'
        return
    if isinstance(item1, dict):
        assert isinstance(item2, dict), f'{path} differs: {item1} != {item2}'
        _check_dict_recursively(item1, item2, path, atol=atol, rtol=rtol)
        return
    if isinstance(item1, (tuple, list)):
        assert isinstance(item2, type(item1)), f'{path} differs: {item1} != {item2}'
        _check_list_recursively(item1, item2, path, atol=atol, rtol=rtol)
        return

    raise NotImplementedError(f'Unsupported item type: {type(item1)}')


def _check_list_recursively(
    list1: Union[Tuple[Any], List[Any]],
    list2: Union[Tuple[Any], List[Any]],
    path: str,
    atol: float,
    rtol: float,
):
    assert len(list1) == len(list2), f'{path} differs: {list1} != {list2}'
    for i, (item1, item2) in enumerate(zip(list1, list2)):
        _check_item(item1, item2, path=f'{path}/{i}', atol=atol, rtol=rtol)


def _check_dict_recursively(dict1: Dict[str, Any], dict2: Dict[str, Any], path: str, atol: float, rtol: float):
    assert len(dict1) == len(dict2), f'{path} differs: {dict1} != {dict2}'
    for k, val1 in dict1.items():
        val2 = dict2[k]
        _check_item(val1, val2, path=f'{path}/{k}', atol=atol, rtol=rtol)
