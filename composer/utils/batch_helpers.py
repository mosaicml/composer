# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers to get items and set items in a batch."""
from __future__ import annotations

from operator import attrgetter, itemgetter
from typing import Any, Callable, Sequence, Union, cast

__all__ = ['batch_get', 'batch_set']


def batch_get(batch: Any, key: Union[str, int, tuple[Callable, Callable], Callable, Any]):
    """Indexes into the batch given the key.

    >>> from composer.utils.batch_helpers import batch_get
    >>> batch_get([1,2,3], 1)
    2
    >>> batch_get({'a':1, 'b':7}, 'b')
    7
    >>> batch_get([{'a':1, 'b':7},{'c':5}], lambda x: x[1]['c'])
    5
    >>> batch_get([{'a':1, 'b':7},{'c':5}], (lambda x: x[1]['c'], lambda x: 10))
    5

    Args:
        batch (Any): An object that contains the input and label of the items in the batch.
            Can be any abritrary type that user creates, but we assume some sort of
            sequence (list, tuple, tensor, array), mapping (dictionary),
            or attribute store (object with data members, namedtuple).
        key (str | int | tuple[Callable, Callable] | Callable | Any, optional): A key to index into the batch or a
                user-specified function to do the extracting. A pair of callables is also
                supported for cases where a get and set function pair are both passed
                (like in Algorithms). The getter is assumed to be the first of the pair.

    Returns:
        The part of the batch specified by the key. This could be any type
            depending on what the batch is composed of.
    """
    # Case 1: key is a tuple of (getter, setter).
    if (isinstance(key, Sequence) and not isinstance(key, str) and _is_key_get_and_set_fn_pair(key)):
        get_fn, _ = key
        return get_fn(batch)

    # Case 2: key is a getter Callable.
    if isinstance(key, Callable):
        return key(batch)

    # Case 3: key some sort of index or key to use to directly extract from the batch.
    try:
        return itemgetter(key)(batch)
    except (IndexError, TypeError):
        try:
            return itemgetter(*key)(batch)
        except TypeError:
            try:
                return attrgetter(cast(str, key))(batch)
            except TypeError:
                return attrgetter(*key)(batch)


def batch_set(batch: Any, key: Union[str, int, tuple[Callable, Callable], Callable, Any], value: Any) -> Any:
    """Indexes into the batch given the key and sets the element at that index to value.

    This is not an in-place operation for batches of type tuple as tuples are not mutable.

    >>> from composer.utils.batch_helpers import batch_set
    >>> batch_set([1,2,3], key=1, value=8)
    [1, 8, 3]
    >>> batch_set({'a':1, 'b':7}, key='b', value=11)
    {'a': 1, 'b': 11}
    >>> def setter(batch, value):
    ...     batch[1]['d'] = value
    ...     return batch
    ...
    >>> batch_set([{'a':1, 'b':7},{'d':3}], key=setter, value=20)
    [{'a': 1, 'b': 7}, {'d': 20}]
    >>> batch_set([{'a':1, 'b':7},{'d':3}], key=(lambda x: x[0]['b'], setter), value=20)
    [{'a': 1, 'b': 7}, {'d': 20}]


    Args:
        batch (Any): An object that contains the input and label of the items in the batch.
            Can be any abritrary type that user creates, but we assume some sort of
            sequence (list, tuple, tensor, array), mapping (dictionary),
            or attribute store (object with data members, namedtuple).
        key (str | int | tuple[Callable, Callable] | Callable | Any, optional): A key to index into the batch or a user-specified function
            to do the setting. A pair of callables is also supported for cases where a get
            and set function pair are both passed (like in Algorithms). The setter is
            assumed to be the second of the pair.
        value (Any): The value that batch[key] or batch.key gets set to.

    Returns:
        batch (Any): updated batch with value set at key.

    """
    # Case 1: key is a tuple of (getter, setter) callables.
    if (isinstance(key, Sequence) and not isinstance(key, str) and _is_key_get_and_set_fn_pair(key)):
        _, set_fn = key
        return set_fn(batch, value)

    # Case 2: key is a callable.
    if isinstance(key, Callable):
        return key(batch, value)

    # Case 4: key is sequence of sub-keys.
    if isinstance(key, Sequence) and not isinstance(key, str):
        return _batch_set_multiple(batch, key, value)

    # Case 5: key is single object, like string or int.
    else:
        return _batch_set(batch, key, value)


def _batch_set(batch: Any, key: Any, value: Any) -> Any:
    """Sets a key value pair in a non-tuple batch."""
    if isinstance(batch, tuple):
        return _batch_set_tuple(batch, key, value)

    try:
        # Check if one can do a __getitem__ before doing a __setitem__ because dicts can
        # do __setitem__ for elements not in the dict and we do not want that.
        # Note for defaultdict and Counter objects, just calling batch[key] for
        # with a new keyword will create a new key, value pair in the object.
        batch[key]
        batch[key] = value

    # The only acceptable TypeErrors are for an object that doesn't have a __setitem__ or a __getitem__,
    # which is TypeError("... object does not support item assignment") and TypeError('.. object is not subscriptable')
    except TypeError as e:
        if 'object does not support item assignment' in str(e) or 'object is not subscriptable' in str(e):
            pass
        else:  # Other type errors should be raised.
            raise e
    else:
        return batch

    try:
        # Make sure batch has key before setting it.
        getattr(batch, key)
        setattr(batch, key, value)

    # If both (setattr or getattr) and __setitem__ raise exceptions then raise both of them.
    except (AttributeError, TypeError) as e:
        raise RuntimeError(
            f'Unable to set key {key} to value {value} on batch {batch}. Please specify a custom set_fn, if necessary.',
        )
    else:
        return batch


def _batch_set_multiple(batch: Any, key: Any, value: Any) -> Any:
    """Sets multiple key value pairs in a non-tuple batch."""
    # Numpy arrays and Torch tensors can take tuples and lists as keys, so try to do a normal
    # __getitem__ call before resulting to list comprehension.
    try:
        # Check if one can do a __getitem__ before doing a __setitem__ because dicts can
        # do __setitem__ for elements not in the dict and we do not want that.
        batch[key]
        batch[key] = value
        return batch
    # Indexing a list with a sequence results in TypeError
    # Indexing an array/tensor with a sequence that is longer than the rank of the array
    # results in an IndexError.
    except (IndexError, TypeError, KeyError):
        pass
    if not hasattr(value, '__len__') or isinstance(value, str):
        raise ValueError(f'value must be a sequence or array or tensor! and not {type(value)}')
    if len(key) != len(value):
        raise ValueError(f'value must be the same length as key ({len(key)}), but it is {len(value)} instead')
    for single_key, single_value in zip(key, value):
        batch = _batch_set(batch, single_key, single_value)
    return batch


def _batch_set_tuple(batch: Any, key: Union[int, str], value: Any) -> Any:
    """Sets key value pairs in tuples and NamedTuples."""
    if hasattr(batch, '_fields'):  # Namedtuple
        if isinstance(key, str):
            batch = batch._replace(**{key: value})
        else:
            batch_list = list(batch)
            batch_list[key] = value
            batch = batch._make(batch_list)

    else:  # Normal tuple.
        batch = list(batch)
        batch[key] = value
        batch = tuple(batch)

    return batch


def _is_key_get_and_set_fn_pair(key):
    if all(callable(key_element) for key_element in key):
        if len(key) == 2:
            return True
        else:
            raise ValueError(f"If key is a sequence of Callables, it should be of length 2' not {len(key)}")
    return False
