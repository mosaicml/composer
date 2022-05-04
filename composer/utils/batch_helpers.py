from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch


def batch_get(batch: Any, key: Any) -> Any:
    """Indexes into the batch given the key.
    Args:
        batch (Any): An object that contains the input and label of the items in the batch.
            Can be any abritrary type that user creates, but we assume some sort of
            sequence (list, tuple, tensor, array), mapping (dictionary),
            or attribute store (object with data members, namedtuple).
        key (BatchKeys): A key to index into the batch. BatchKeys is of type Union[BatchKey Sequence[BatchKey]]
            and BatchKey is of type Union[int, str, slice].

    Returns:
        The part of the batch specified by the key. This could be any type depending on
            what the batch is composed of.
    """
    if isinstance(key, Sequence) and not isinstance(key, str):
        return _batch_get_multiple(batch, key)
    else:
        return _batch_get(batch, key)


def _batch_get(batch: Any, key: Any) -> Any:
    if isinstance(batch, Tuple):
        return _batch_get_tuple(batch, key)
    elif isinstance(batch, Mapping) or (isinstance(batch, Sequence) and not isinstance(batch, str)):
        value = batch[key]  # int or slice for sequence or string for dict.
    elif isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
        value = batch[key]
    else:
        value = getattr(batch, key)
    return value


def _batch_get_multiple(batch: Any, key: Any):
    return [_batch_get(batch, k) for k in key]


def batch_set(batch: Any, key: Any, value: Any) -> Any:
    """Indexes into the batch given the key and sets the element at that index to value.
    Args:
        batch (Any): An object that contains the input and label of the items in the batch.
            Can be any abritrary type that user creates, but we assume some sort of
            sequence (list, tuple, tensor, array), mapping (dictionary),
            or attribute store (object with data members, namedtuple).
        key (BatchKeys): A key to index into the batch. BatchKeys is of type Union[BatchKey Sequence[BatchKey]]
            and BatchKey is of type Union[int, str, slice].
        value (Any): The value that batch[key] or batch.key gets set to.
    Returns:
        batch (Any): updated batch with value set at key.
    """
    if isinstance(key, Sequence) and not isinstance(key, str):
        return _batch_set_multiple(batch, key, value)
    else:
        return _batch_set(batch, key, value)


def _batch_set(batch: Any, key: Any, value: Any) -> Any:
    """Sets a key value pair in a non-tuple batch."""
    if isinstance(batch, tuple):
        batch = _batch_set_tuple(batch, key, value)
    elif isinstance(batch, Mapping) or (isinstance(batch, Sequence) and not isinstance(batch, str)):
        batch[key] = value  # type: ignore
    elif isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
        batch[key] = value
    else:
        setattr(batch, key, value)
    return batch


def _batch_set_multiple(batch: Any, key: Any, value: Any) -> Any:
    """Sets multiple key value pairs in a non-tuple batch."""
    if len(key) != len(value):
        raise ValueError(f'value must be the same length as key ({len(key)}), but it is f{len(value)} instead')
    for single_key, single_value in zip(key, value):
        batch = _batch_set(batch, single_key, single_value)
    return batch


def _batch_set_tuple(batch: Any, key: Any, value: Any) -> Any:
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


def _batch_get_tuple(batch: Any, key: Any) -> Any:
    """"Gets keys in tuples and NamedTuples."""
    is_named_tuple = hasattr(batch, '_fields')
    if is_named_tuple and isinstance(key, str):  # NamedTuple w/ named key
        value = getattr(batch, key)
    else:  # Normal tuple or namedtuple with int key.
        value = batch[key]

    return value
