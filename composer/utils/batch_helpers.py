# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, Sequence, Union

__all__ = ['batch_get', 'batch_set']


def batch_get(batch: Any, key: Optional[Any] = None, get_fn: Optional[Callable[[Any], Any]] = None) -> Any:
    """Indexes into the batch given the key.

    >>> from composer.utils.batch_helpers import batch_get
    >>> batch_get([1,2,3], 1)
    2
    >>> batch_get({'a':1, 'b':7}, 'b')
    7
    >>> batch_get([{'a':1, 'b':7},{'c':5}], get_fn=lambda x: x[1]['c'])
    5

    Args:
        batch (Any): An object that contains the input and label of the items in the batch.
            Can be any abritrary type that user creates, but we assume some sort of
            sequence (list, tuple, tensor, array), mapping (dictionary),
            or attribute store (object with data members, namedtuple).
        key (Any): A key to index into the batch. Key is optional if get_fn is supplied.
        get_fn (Callable): A user-specified function to do the extracting. 
            get_fn is optional if key is supplied.

    Returns:
        The part of the batch specified by the key or the get_fn. This could be any type 
            depending on what the batch is composed of.
    
    Raises:
        ValueError if key is unset and get_fn is unset or if both are set.
    """
    if key is None and get_fn is None:
        raise ValueError("key or get_fn must be specified and neither were!")
    if key is not None and get_fn is not None:
        raise ValueError("key and get_fn were both set. Only one can be set!")
    if get_fn is not None:
        return get_fn(batch)
    if isinstance(key, Sequence) and not isinstance(key, str):
        return _batch_get_multiple(batch, key)
    else:
        return _batch_get(batch, key)


def _batch_get(batch: Any, key: Any) -> Any:
    if isinstance(batch, tuple):
        return _batch_get_tuple(batch, key)

    try:
        return batch[key]

    # The only acceptable TypeError is for an object that doesn't have a __getitem__,
    # which is TypeError("... object is not subscriptable").
    except TypeError as e:
        if 'object is not subscriptable' in str(e):
            pass
        else:
            raise e

    try:
        return getattr(batch, key)

    # If both getattr and __getitem__ result in exceptions then raise both of them.
    except (AttributeError, TypeError):
        raise RuntimeError(
            f"Unable extract key {key} from batch {batch}. Please specify a custom get_fn, if necessary.")


def _batch_get_multiple(batch: Any, key: Any):
    # Numpy arrays and Torch tensors can take tuples and lists as keys.
    try:
        return batch[key]
    # Indexing a list with a sequence results in TypeError
    # Indexing an array/tensor with a sequence that is longer than the rank of the array
    # results in an IndexError.
    # Indexing a dict with a tuple results in a key error.
    except (IndexError, TypeError, KeyError):
        pass
    return [_batch_get(batch, k) for k in key]


def batch_set(batch: Any,
              *,
              key: Optional[Any] = None,
              value: Any,
              set_fn: Optional[Callable[[Any, Any], Any]] = None) -> Any:
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
    >>> batch_set([{'a':1, 'b':7},{'d':3}], value=20, set_fn=setter)
    [{'a': 1, 'b': 7}, {'d': 20}]

    Args:
        batch (Any): An object that contains the input and label of the items in the batch.
            Can be any abritrary type that user creates, but we assume some sort of
            sequence (list, tuple, tensor, array), mapping (dictionary),
            or attribute store (object with data members, namedtuple).
        key (Any): A key to index into the batch.
        value (Any): The value that batch[key] or batch.key gets set to.
        set_fn (Callable): A user-specified function to do the setting. set_fn is optional if key and 
            value are supplied. set_fn must return the updated batch.

    Returns:
        batch (Any): updated batch with value set at key.

    Raises:
        ValueError if:
            * key and set_fn are both unset
            * key and set_fn are both set
    """
    if key is None and set_fn is None:
        raise ValueError("key or set_fn must be specified and neither were!")
    if key is not None and set_fn is not None:
        raise ValueError("key and set_fn were both set. Only one can be set!")
    if set_fn:
        return set_fn(batch, value)
    if isinstance(key, Sequence) and not isinstance(key, str):
        return _batch_set_multiple(batch, key, value)
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
            f"Unable to set key {key} to value {value} on batch {batch}. Please specify a custom set_fn, if necessary.")
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
    """"Sets key value pairs in tuples and NamedTuples."""
    if hasattr(batch, '_fields'):  # NamedTuple
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


def _batch_get_tuple(batch: Any, key: Union[int, str]) -> Any:
    """"Gets keys in tuples and NamedTuples."""
    is_named_tuple = hasattr(batch, '_fields')
    if is_named_tuple and isinstance(key, str):  # NamedTuple w/ named key
        value = getattr(batch, key)
    else:  # Normal tuple or namedtuple with int key.
        value = batch[key]

    return value
