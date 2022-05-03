from collections import OrderedDict
from typing import NamedTuple

import numpy as np
import pytest
import torch

from composer.utils.batch_helpers import batch_get, batch_set

my_list = [3, 4, 5, 6, 7, 8, 9, 10]

keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


class myClass(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


list_types = [type(element) for element in my_list]
my_named_tuple = NamedTuple('nt', **dict(zip(keys, list_types)))


@pytest.fixture(scope="module",
                params=[my_list,
                        tuple(my_list),
                        torch.tensor(my_list),
                        np.asarray(my_list),
                        my_named_tuple(*my_list)])
def example_sequence(request):
    return request.param


# All key value pair data structures that have a __getitem__ function thats takes str.
@pytest.fixture(scope="module",
                params=[
                    dict(zip(keys, my_list)),
                    myClass(**dict(zip(keys, my_list))),
                    my_named_tuple(*my_list),
                    OrderedDict(**dict(zip(keys, my_list)))
                ])
def example_map(request):
    return request.param


# # All key-value pair data structures that are mutable.
# @pytest.fixture(
#     scope="module",
#     params=[dict(zip(keys, my_list)),
#             myClass(**dict(zip(keys, my_list))),
#             OrderedDict(**dict(zip(keys, my_list)))])
# def example_map(request):
#     return request.param


# Test whether sequences can be indexed by an int.
def test_int_key(example_sequence, key=2, expected=5):
    assert batch_get(example_sequence, key) == expected


# Test whether kv pair data structures can be indexed by a str.
def test_str_key(example_map, key='d', expected=6):
    assert batch_get(example_map, key) == expected


# Test whether sequences can be indexed by a sequence of ints.
def test_sequence_of_ints_key(example_sequence, key=[2, 5, 7], expected=[5, 8, 10]):
    assert list(batch_get(example_sequence, key)) == expected


# Test whether kv pair data structures can be indexed by a sequence of strings.
def test_sequence_of_strs_key(example_map, key=['c', 'f'], expected=[5, 8]):
    assert list(batch_get(example_map, key)) == expected


# Test whether sequences can be indexed by a slice object.
def test_slice_key(example_sequence, key=slice(1, 6, 2), expected=[4, 6, 8]):
    assert list(batch_get(example_sequence, key)) == expected


# Test whether sequences can be indexed by a sequence of slice objects.
@pytest.mark.parametrize('batch,key,expected', [(my_list, [slice(1, 4), slice(5, 7)], [[4, 5, 6], [8, 9]]),
                                                (tuple(my_list), [slice(1, 4), slice(5, 7)], [(4, 5, 6), (8, 9)]),
                                                (my_named_tuple(*my_list), [slice(1, 4), slice(5, 7)], [(4, 5, 6),
                                                                                                        (8, 9)])])
def test_seq_of_slices_key(batch, key, expected):
    assert batch_get(batch, key) == expected


# Test whether arrays and tensors can be indexed by a sequence of slice objects.
@pytest.mark.parametrize('batch,key,expected', [(torch.tensor(my_list), [slice(1, 4), slice(5, 7)], [
    torch.tensor([4, 5, 6]), torch.tensor([8, 9])
]), (np.asarray(my_list), [slice(1, 4), slice(5, 7)], [np.asarray([4, 5, 6]), np.asarray([8, 9])])])
def test_seq_of_slices_key_for_tensors_and_arrays(batch, key, expected):
    for actual, expectation in zip(batch_get(batch, key), expected):
        assert all(actual == expectation)


@pytest.fixture
def example_list():
    return [3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def example_tensor():
    return torch.tensor([3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def example_array():
    return np.asarray([3, 4, 5, 6, 7, 8, 9, 10])


# Test whether lists can be set using batch_set.
@pytest.mark.parametrize('key,value', [(1, 2), (3, 7), ([2, 5], [11, 13]), (slice(1, 6, 2), [-1, -3, -5]),
                                       ([slice(0, 3, 1), slice(4, 7, 1)], [[10, 11, 12], [13, 14, 15]])])
def test_batch_set_list(example_list, key, value):
    batch_set(example_list, key, value)
    assert batch_get(example_list, key) == value


# Test whether tensors can be set using batch_set.
@pytest.mark.parametrize('key,value', [(1, torch.tensor(2)), (3, torch.tensor(7)), ([2, 5], torch.tensor([11, 13])),
                                       (slice(1, 6, 2), torch.tensor([-1, -3, -5]))])
def test_batch_set_tensor(example_tensor, key, value):
    batch_set(example_tensor, key, value)
    assert torch.equal(batch_get(example_tensor, key), value)


# Test whether tensors can be set using batch_set with a list of slices.
def test_batch_set_tensor_list_of_slices(example_tensor,
                                         key=[slice(0, 3, 1), slice(4, 7, 1)],
                                         value=[torch.tensor([10, 11, 12]),
                                                torch.tensor([13, 14, 15])]):
    batch_set(example_tensor, key, value)
    for actual, expectation in zip(batch_get(example_tensor, key), value):
        assert torch.equal(actual, expectation)


# Test whether arrays can be set using batch_set.
@pytest.mark.parametrize('key,value', [(1, np.asarray(2)), (3, np.asarray(7)), ([2, 5], np.asarray([11, 13])),
                                       (slice(1, 6, 2), np.asarray([-1, -3, -5]))])
def test_batch_set_array(example_array, key, value):
    batch_set(example_array, key, value)
    assert np.all(batch_get(example_array, key) == value)


# Test whether arrays can be set using batch_set with a list of slices.
def test_batch_set_array_list_of_slices(example_array,
                                        key=[slice(0, 3, 1), slice(4, 7, 1)],
                                        value=[np.asarray([10, 11, 12]),
                                               np.asarray([13, 14, 15])]):
    batch_set(example_array, key, value)
    for actual, expectation in zip(batch_get(example_array, key), value):
        assert np.all(actual == expectation)


# Test whether mutable key value data structures can be set using batch_set.
@pytest.mark.parametrize('key,value', [('b', -10), ('c', -20), (['d', 'e'], [100, 101])])
def test_batch_set_map(example_map, key, value):
    new_batch = batch_set(example_map, key, value)
    assert batch_get(new_batch, key) == value
