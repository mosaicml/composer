from typing import Any, Sequence, NamedTuple, List, Tuple

from composer.core.types import BatchKeys


def batch_get(batch: Any, key: BatchKeys) -> Any:
    """Indexes into the batch given the key.

    Tries all the common combination of batch and key pairs:
        * int for sequence
        * slice for sequence
        * string for dict
        * string atribute for object or named_tuple
        * sequence of ints for sequence
        * sequence of slices for sequence
        * sequence of keys for dict
        * sequence of attribute names for object or named_tuple

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

    Raises:
        RuntimeError if the batch can't be indexed by the key.
    """
    try:
        return batch[key]  # int or slice for sequence or string for dict.
    except:
        pass
    try:
        return [batch[k] for k in key]  # Sequence of ints, slices, or strings.
    except:
        pass
    try:
        return getattr(batch, key)  # Attribute for object or named tuple.
    except:
        pass
    try:
        return [getattr(batch, k) for k in key]  # Sequence of attributes for object or named tuple.
    except:
        raise RuntimeError(f"Batch object can't be indexed by nor has attribute {key}")


def batch_set(batch: Any, key: BatchKeys, value: Any) -> Any:
    """Indexes into the batch given the key and sets the element at that index to value.

    Tries all the common combination of batch and key pairs:
        * int for sequence
        * slice for sequence
        * string for dict
        * string atribute for object or named_tuple
        * sequence of ints for sequence
        * sequence of slices for sequence
        * sequence of keys for dict
        * sequence of attribute names for object or named_tuple

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
    Raises:
        RuntimeError if the batch can't be indexed by the key or batch can't be set at that key.
    """
    if isinstance(batch, tuple):
        return _batch_set_tuple(batch, key, value)
    else:
        try:
            batch[key] = value  # int or slice for sequence or string for dict.
            return batch
        except:
            pass
        try:
            for k, v in zip(key, value):  # Sequence of ints, slices, or strings.
                batch[k] = v
            return batch
        except:

            pass
        try:
            setattr(batch, key, value)  # Attribute for object or named tuple.
            return batch
        except:
            pass
        try:
            for k, v in zip(key, value):  # Sequence of attributes for object or named tuple.
                setattr(batch, k, v)
            return batch
        except:
            raise RuntimeError(f"Batch object of type {str(type(batch))} can't be set at index or attribute {key}")


def _batch_set_tuple(batch, key, value):
    """"Sets key value pairs in tuples and NamedTuples."""
    if hasattr(batch, '_fields'): # NamedTuple
        if isinstance(key, (List, Tuple)):
            for k,v in zip(key, value):
                if isinstance(k, str):
                    batch = batch._replace(**{k:v})
                else: # Has to be int because you cannot have a named tuple with keys other than str.
                    batch_list = list(batch)
                    batch_list[k] = v
                    batch = batch._make(batch_list)
        else:
            if isinstance(key, str):
                batch = batch._replace(**{key:value})
            else:
                batch_list = list(batch)
                batch_list[key] = value
                batch = batch._make(batch_list)
        return batch
    
    else:     # Normal tuple.
        if isinstance(key, (List, Tuple)):
            batch = list(batch)
            for k,v in zip(key, value):
                batch[k] = v

        else:
            batch = list(batch)
            batch[key] = value

        return tuple(batch)