# Copyright 2021 MosaicML. All Rights Reserved.

# To keep the typing organized for this file, see iter_helpers.pyi
# All typing annotations are in there
# All methods signatures must be defined in there.


def map_collection(collection, map_fn):
    """map takes a single element, or a collection of elements, and applies `map_fn` on the element (or each
    element when `maybe_tuple` is a collection).
    It returns the result of `map_fn` in the same data format as `collection` -- i.e. dicts are returned as dicts

    Args:
        collection: The element, or a tuple of elements
        map_fn: A function to invoke on each element

    Returns:
        When `collection` is a single element, it returns the result of `map_fn` on `maybe_tuple`.
        When `collection` is a collection, it returns the results of `map_fn` for each element in
        `collection` in the same data format as the original `collection`
    """
    if isinstance(collection, (tuple, list)):
        return type(collection)(map_fn(x) for x in collection)
    if isinstance(collection, dict):
        return {k: map_fn(v) for k, v in collection.items()}
    return map_fn(collection)


def ensure_tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, dict):
        return tuple(x.values())
    return (x,)


def zip_collection(item, *others):
    if isinstance(item, (tuple, list, dict)):
        others = [ensure_tuple(other) for other in others]
    else:
        item = [item]
        others = [[other] for other in others]
    return zip(item, *others)
