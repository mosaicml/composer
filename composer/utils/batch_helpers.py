from typing import Any, Sequence

__all__ = ['batch_get', 'batch_set']


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
    if isinstance(batch, tuple):
        return _batch_get_tuple(batch, key)

    exceptions = []
    try:
        value = batch[key]

    # The only acceptable TypeError is for an object that doesn't have a __getitem__,
    # which is TypeError("... object is not subscriptable").
    except TypeError as e:
        if 'object is not subscriptable' in str(e):
            exceptions.append(e)
            pass
        else:
            raise e
    else:
        return value

    try:
        value = getattr(batch, key)

    # If both getattr and __getitem__ result in exceptions then raise both of them.
    except Exception as e:  # AttributeError, TypeError.
        exceptions.append(e)
        raise Exception(exceptions)
    else:
        return value


def _batch_get_multiple(batch: Any, key: Any):
    return [_batch_get(batch, k) for k in key]


def batch_set(batch: Any, key: Any, value: Any) -> Any:
    """Indexes into the batch given the key and sets the element at that index to value.

    This is not an in-place operation for batches of type tuple as tuples are not mutable.
    
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
        return _batch_set_tuple(batch, key, value)

    exceptions = []
    try:
        batch[key] = value

    # The only acceptable TypeError is for an object that doesn't have a __setitem__,
    # which is TypeError("... object does not support item assignment").
    except TypeError as e:
        if 'object does not support item assignment' in str(e):
            exceptions.append(e)
            pass
        else:
            raise e
    else:
        return batch

    try:
        setattr(batch, key, value)

    # If both setattr and __setitem__ raise exceptions then raise both of them.
    except Exception as e:  # AttributeError, TypeError.
        exceptions.append(e)
        raise Exception(exceptions)
    else:
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
