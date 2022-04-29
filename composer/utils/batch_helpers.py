from typing import Any

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
