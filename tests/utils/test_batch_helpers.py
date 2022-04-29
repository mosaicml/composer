from collections import namedtuple

import numpy as np
import pytest
import torch
from matplotlib import get_backend

from composer.utils.batch_helpers import batch_get

my_list = [3, 4, 5, 6, 7, 8, 9, 10]

keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


class myClass(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


my_named_tuple = namedtuple('nt', keys)


@pytest.fixture(scope="module",
                params=[my_list,
                        tuple(my_list),
                        torch.tensor(my_list),
                        np.asarray(my_list),
                        my_named_tuple(*my_list)])
def example_sequence(request):
    return request.param


@pytest.fixture(scope="module",
                params=[dict(zip(keys, my_list)),
                        myClass(**dict(zip(keys, my_list))),
                        my_named_tuple(*my_list)])
def example_map(request):
    return request.param


def test_int_key(example_sequence, key=2, expected=5):
    assert batch_get(example_sequence, key) == expected


def test_str_key(example_map, key='d', expected=6):
    assert batch_get(example_map, key) == expected


def test_sequence_of_ints_key(example_sequence, key=[2, 5, 7], expected=[5, 8, 10]):
    assert list(batch_get(example_sequence, key)) == expected


def test_sequence_of_strs_key(example_map, key=['c', 'f'], expected=[5, 8]):
    assert list(batch_get(example_map, key)) == expected


def test_slice_key(example_sequence, key=slice(1, 6, 2), expected=[4, 6, 8]):
    assert list(batch_get(example_sequence, key)) == expected


@pytest.mark.parametrize('batch,key,expected', [(my_list, [slice(1, 4), slice(5, 7)], [[4, 5, 6], [8, 9]]),
                                                (tuple(my_list), [slice(1, 4), slice(5, 7)], [(4, 5, 6), (8, 9)]),
                                                (my_named_tuple(*my_list), [slice(1, 4), slice(5, 7)], [(4, 5, 6),
                                                                                                        (8, 9)])])
def test_seq_of_slices_key(batch, key, expected):
    assert batch_get(batch, key) == expected


@pytest.mark.parametrize('batch,key,expected', [(torch.tensor(my_list), [slice(1, 4), slice(5, 7)], [
    torch.tensor([4, 5, 6]), torch.tensor([8, 9])
]), (np.asarray(my_list), [slice(1, 4), slice(5, 7)], [np.asarray([4, 5, 6]), np.asarray([8, 9])])])
def test_seq_of_slices_key_for_tensors_and_arrays(batch, key, expected):
    for actual, expectation in zip(batch_get(batch, key), expected):
        assert all(actual == expectation)
