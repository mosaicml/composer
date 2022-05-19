from collections import ChainMap, Counter, OrderedDict, defaultdict, deque
from typing import NamedTuple

import numpy as np
import pytest
import torch

from composer.utils.batch_helpers import batch_get, batch_set

my_list = [3, 4, 5, 6, 7, 8, 9, 10]

keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


class MyClass(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


list_types = [type(element) for element in my_list]
my_named_tuple = NamedTuple('nt', **dict(zip(keys, list_types)))
counter_list = []
for char, num in zip(keys, my_list):
    counter_list.extend(num * [char])


@pytest.fixture(scope="module",
                params=[
                    my_list,
                    tuple(my_list),
                    torch.tensor(my_list),
                    np.asarray(my_list),
                    deque(my_list),
                    my_named_tuple(*my_list)
                ])
def example_sequence(request):
    return request.param


@pytest.fixture(scope="module",
                params=[my_list,
                        tuple(my_list),
                        torch.tensor(my_list),
                        np.asarray(my_list),
                        my_named_tuple(*my_list)])
def example_deque_less_sequence(request):
    return request.param


# All key value pair data structures that have a __getitem__ function thats takes str.
@pytest.fixture(scope="module",
                params=[
                    dict(zip(keys, my_list)),
                    MyClass(**dict(zip(keys, my_list))),
                    my_named_tuple(*my_list),
                    defaultdict(list, **dict(zip(keys, my_list))),
                    ChainMap(dict(zip(keys, my_list)), dict(a=7, j=3)),
                    Counter(counter_list),
                    OrderedDict(**dict(zip(keys, my_list)))
                ])
def example_map(request):
    return request.param


# Test whether sequences can be indexed by an int.
def test_int_key(example_sequence, key=2, expected=5):
    assert batch_get(example_sequence, key) == expected


# Test whether kv pair data structures can be indexed by a str.
def test_str_key(example_map, key='d', expected=6):
    assert batch_get(example_map, key) == expected


# Test whether sequences can be indexed by a sequence of ints.
def test_sequence_of_ints_key(example_sequence):
    key = [2, 5, 7]
    expected = [5, 8, 10]
    assert list(batch_get(example_sequence, key)) == expected


# Test whether kv pair data structures can be indexed by a sequence of strings.
def test_sequence_of_strs_key(example_map):
    key = ['c', 'f']
    expected = [5, 8]
    assert list(batch_get(example_map, key)) == expected


# Test whether sequences can be indexed by a slice object.
def test_slice_key(example_deque_less_sequence):
    key = slice(1, 6, 2)
    expected = [4, 6, 8]
    assert list(batch_get(example_deque_less_sequence, key)) == expected


# Test whether sequences can be indexed by a sequence of slice objects.
@pytest.mark.parametrize('batch,key,expected', [(my_list, [slice(1, 4), slice(5, 7)], [[4, 5, 6], [8, 9]]),
                                                (tuple(my_list), [slice(1, 4), slice(5, 7)], [(4, 5, 6), (8, 9)]),
                                                (my_named_tuple(*my_list), [slice(1, 4), slice(5, 7)], [(4, 5, 6),
                                                                                                        (8, 9)])])
def test_seq_of_slices_key(batch, key, expected):
    assert batch_get(batch, key) == expected


@pytest.fixture
def example_list():
    return my_list


@pytest.fixture
def example_tuple():
    return tuple(my_list)


@pytest.mark.parametrize('batch', [my_list, tuple(my_list), my_named_tuple(*my_list)])
def test_batch_set_sequence_int_key(batch, key=3, value=23):
    new_batch = batch_set(batch, key=key, value=value)
    assert batch_get(new_batch, key) == value


# Test whether lists can be set using batch_set.
@pytest.mark.parametrize('key,value', [([2, 5], [11, 13]), ([2, 5], (11, 13)), (slice(1, 6, 2), [-1, -3, -5]),
                                       ([slice(0, 3, 1), slice(4, 7, 1)], [[10, 11, 12], [13, 14, 15]])])
def test_batch_set_list(example_list, key, value):
    new_batch = batch_set(example_list, key=key, value=value)
    assert tuple(batch_get(new_batch, key)) == tuple(value)


# Test whether tuples can be set using batch_set.
@pytest.mark.parametrize('key,value', [([2, 5], [11, 13]), ([2, 5], (11, 13)), (slice(1, 6, 2), [-1, -3, -5]),
                                       ([slice(0, 3, 1), slice(4, 7, 1)], [(10, 11, 12), (13, 14, 15)])])
def test_batch_set_tuple(example_tuple, key, value):
    new_batch = batch_set(example_tuple, key=key, value=value)
    assert tuple(batch_get(new_batch, key)) == tuple(value)


# Test whether tuples can be set using batch_set.
@pytest.mark.parametrize('key,value', [([2, 5], [11, 13]), ([2, 5], (11, 13)), (slice(1, 6, 2), [-1, -3, -5]),
                                       ([slice(0, 3, 1), slice(4, 7, 1)], [(10, 11, 12), (13, 14, 15)])])
def test_batch_set_named_tuple(key, value):
    batch = my_named_tuple(*my_list)
    new_batch = batch_set(batch, key=key, value=value)
    assert tuple(batch_get(new_batch, key)) == tuple(value)


# Test whether mutable key value data structures can be set using batch_set.
@pytest.mark.parametrize('key,value', [('b', -10), ('c', -20), (['d', 'e'], [100, 101])])
def test_batch_set_map(example_map, key, value):
    new_batch = batch_set(example_map, key=key, value=value)
    assert batch_get(new_batch, key) == value


##### Arrays and Tensors.


@pytest.fixture
def example_tensor():
    return torch.tensor([3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def example_array():
    return np.asarray([3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def example_2D_array():
    return np.arange(12).reshape(4, 3)


@pytest.fixture
def example_2D_tensor():
    return torch.arange(12).reshape(4, 3)


@pytest.fixture(scope="module", params=[np.arange(12).reshape(4, 3), torch.arange(12).reshape(4, 3)])
def example_2D_array_tensor(request):
    return request.param


# Test whether arrays and tensors can be indexed by a sequence of int objects.
@pytest.mark.parametrize('batch,key,expected', [(torch.tensor(my_list), [1, 4], torch.tensor([4, 7])),
                                                (torch.tensor(my_list), torch.tensor([1, 4]), torch.tensor([4, 7])),
                                                (torch.tensor(my_list), np.asarray([1, 4]), torch.tensor([4, 7])),
                                                (np.asarray(my_list), [1, 4], np.asarray([4, 7])),
                                                (np.asarray(my_list), torch.tensor([1, 4]), np.asarray([4, 7])),
                                                (np.asarray(my_list), np.asarray([1, 4]), np.asarray([4, 7]))])
def test_batch_get_seq_key_for_1D_tensors_and_arrays(batch, key, expected):
    assert all(batch_get(batch, key) == expected)


# Test whether arrays and tensors can be indexed by a sequence of slice objects.
@pytest.mark.parametrize('batch,key,expected', [(torch.tensor(my_list), [slice(1, 4), slice(5, 7)], [
    torch.tensor([4, 5, 6]), torch.tensor([8, 9])
]), (np.asarray(my_list), [slice(1, 4), slice(5, 7)], [np.asarray([4, 5, 6]), np.asarray([8, 9])])])
def test_batch_get_seq_of_slices_key_for_1D_tensors_and_arrays(batch, key, expected):
    for actual, expectation in zip(batch_get(batch, key), expected):
        assert all(actual == expectation)


# Test whether tensors can be set using batch_set.
@pytest.mark.parametrize('key,value', [(1, torch.tensor(2)), ([2, 5], torch.tensor([11, 13])),
                                       (np.asarray([2, 5]), torch.tensor([11, 13])),
                                       (torch.tensor([2, 5]), torch.tensor([11, 13])),
                                       (slice(1, 6, 2), torch.tensor([-1, -3, -5]))])
def test_batch_set_1D_tensor(example_tensor, key, value):
    new_batch = batch_set(example_tensor, key=key, value=value)
    assert torch.equal(torch.tensor(batch_get(new_batch, key)), value)


# Test whether arrays can be set using batch_set.
@pytest.mark.parametrize('key,value', [(1, np.asarray(2)), ([2, 5], np.asarray([11, 13])),
                                       (np.asarray([2, 5]), np.asarray([11, 13])),
                                       (torch.tensor([2, 5]), np.asarray([11, 13])),
                                       (slice(1, 6, 2), np.asarray([-1, -3, -5]))])
def test_batch_set_1D_array(example_array, key, value):
    new_batch = batch_set(example_array, key=key, value=value)
    assert np.all(batch_get(new_batch, key) == value)


# Test whether tensors can be set using batch_set with a list of slices.
def test_batch_set_1D_tensor_list_of_slices_key(example_tensor):
    key = [slice(0, 3, 1), slice(4, 7, 1)]
    value = [torch.tensor([10, 11, 12]), torch.tensor([13, 14, 15])]
    new_batch = batch_set(example_tensor, key=key, value=value)
    for actual, expectation in zip(batch_get(new_batch, key), value):
        assert torch.equal(actual, expectation)


# Test whether arrays can be set using batch_set with a list of slices.
def test_batch_set_1D_array_list_of_slices_key(example_array):
    key = [slice(0, 3, 1), slice(4, 7, 1)]
    value = [np.asarray([10, 11, 12]), np.asarray([13, 14, 15])]
    new_batch = batch_set(example_array, key=key, value=value)
    for actual, expectation in zip(batch_get(new_batch, key), value):
        assert np.all(actual == expectation)


@pytest.mark.parametrize('key,expected', [((1, 2), 5)])
def test_batch_get_2D_array_tensor_2D_tuple_key(example_2D_array_tensor, key, expected):
    actual = batch_get(example_2D_array_tensor, key)
    assert int(actual) == expected


@pytest.mark.parametrize('key,expected', [([1, 2], [[3, 4, 5], [6, 7, 8]]),
                                          (np.asarray([1, 2]), [[3, 4, 5], [6, 7, 8]]),
                                          (torch.tensor([1, 2]), [[3, 4, 5], [6, 7, 8]])])
def test_batch_get_2D_array_tensor_2D_key(example_2D_array_tensor, key, expected):
    actual = batch_get(example_2D_array_tensor, key)
    assert actual.tolist() == expected


@pytest.mark.parametrize('key,expected', [([slice(2, 4), slice(1, 3)], [[7, 8], [10, 11]])])
def test_batch_get_2D_array_tensor_2D_slice_key(example_2D_array_tensor, key, expected):
    actual = batch_get(example_2D_array_tensor, key)
    assert actual.tolist() == expected


@pytest.mark.parametrize('key,value', [((1, 2), 6)])
def test_batch_set_2D_array_and_tensor_2D_tuple_key(example_2D_array_tensor, key, value):
    batch = batch_set(example_2D_array_tensor, key=key, value=value)
    assert batch_get(batch, key) == value


@pytest.mark.parametrize('key,value', [([1, 2], torch.tensor([[3, 6, 9], [6, 12, 18]])),
                                       (np.asarray([1, 2]), torch.tensor([[3, 6, 9], [6, 12, 18]])),
                                       (torch.tensor([1, 2]), torch.tensor([[3, 6, 9], [6, 12, 18]]))])
def test_batch_set_2D_tensor_2D_seq_key(example_2D_tensor, key, value):
    new_batch = batch_set(example_2D_tensor, key=key, value=value)
    assert torch.equal(batch_get(new_batch, key), value)


def test_batch_set_2D_tensor_list_of_slices(example_2D_tensor):
    key = [slice(2, 4), slice(1, 3)]
    value = torch.tensor([[7, 14], [10, 20]])
    new_batch = batch_set(example_2D_tensor, key=key, value=value)
    assert torch.equal(batch_get(new_batch, key), value)


@pytest.mark.parametrize('key,value', [([1, 2], np.asarray([[3, 6, 9], [6, 12, 18]])),
                                       (np.asarray([1, 2]), np.asarray([[3, 6, 9], [6, 12, 18]])),
                                       (torch.tensor([1, 2]), np.asarray([[3, 6, 9], [6, 12, 18]]))])
def test_batch_set_2D_array_2D_seq_key(example_2D_array, key, value):
    new_batch = batch_set(example_2D_array, key=key, value=value)
    assert np.all(np.equal(batch_get(new_batch, key), value))


def test_batch_set_2D_array_list_of_slices(example_2D_array):
    key = [slice(2, 4), slice(1, 3)]
    value = np.asarray([[7, 14], [10, 20]])
    new_batch = batch_set(example_2D_array, key=key, value=value)
    assert np.all(np.equal(batch_get(new_batch, key), value))


@pytest.fixture
def example_complicated_object():
    return [dict(a=[1, 2], b=[2, 4]), dict(c=[3, 6], d=[5, 7])]


@pytest.fixture
def example_get_callable():

    def my_get_callable(batch):
        return batch[1]['d'][0]

    return my_get_callable


@pytest.fixture
def example_set_callable():

    def my_set_callable(batch, value):
        batch[1]['d'][0] = value
        return batch

    return my_set_callable


def test_batch_get_callable(example_complicated_object, example_get_callable):
    assert batch_get(example_complicated_object, get_fn=example_get_callable) == 5


def test_batch_set_callable(example_complicated_object, example_set_callable, example_get_callable):
    new_batch = batch_set(example_complicated_object, value=11, set_fn=example_set_callable)
    assert batch_get(new_batch, get_fn=example_get_callable) == 11


def test_batch_get_errors(example_complicated_object, example_get_callable):
    # Neither key nor gert_fn specified
    with pytest.raises(ValueError):
        batch_get(example_complicated_object)

    # key and get_fn set.
    with pytest.raises(ValueError):
        batch_get(example_complicated_object, key=1, get_fn=example_get_callable)


def test_batch_set_errors(example_complicated_object, example_set_callable):
    # key and set_fn unset.
    with pytest.raises(ValueError):
        batch_set(example_complicated_object, value=11)

    # key and set_fn set.
    with pytest.raises(ValueError):
        batch_set(example_complicated_object, key=1, value=11, set_fn=example_set_callable)


def test_set_with_mismatched_key_values(example_list):
    with pytest.raises(ValueError):
        batch_set(example_list, key=[1, 3, 5], value=[1, 2])
    with pytest.raises(ValueError):
        batch_set(example_list, key=[1, 3, 5], value=1)


# It's almost impossible to stop Counter and defaultdict from adding
# new items, so we don't incliude them here.
@pytest.mark.parametrize('batch', [
    dict(zip(keys, my_list)),
    MyClass(**dict(zip(keys, my_list))),
    my_named_tuple(*my_list),
    ChainMap(dict(zip(keys, my_list)), dict(a=7, j=3)),
    OrderedDict(**dict(zip(keys, my_list)))
])
def test_batch_set_with_new_key_fails(batch):
    with pytest.raises(Exception):
        batch_set(batch, key='key_that_is_certainly_not_present', value=5)
