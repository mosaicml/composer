# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# To keep the typing organized for this file, see iter_helpers.pyi
# All typing annotations are in there
# All methods signatures must be defined in there.

"""Utilities for iterating over collections."""
from __future__ import annotations

import collections.abc
import io
from typing import Any


def map_collection(collection, map_fn) -> Any:
    """Applies ``map_fn`` on each element in ``collection``.

    * If ``collection`` is a tuple or list of elements, ``map_fn`` is applied on each element,
      and a tuple or list, respectively, containing mapped values is returned.
    * If ``collection`` is a dictionary, ``map_fn`` is applied on each value, and a dictionary
      containing the mapped values is returned.
    * If ``collection`` is ``None``, ``None`` is returned.
    * If ``collection`` is a single element, the result of applying ``map_fn`` on it is returned.

    Args:
        collection: The element, or a tuple of elements.
        map_fn: A function to invoke on each element.

    Returns:
        Collection: The result of applying ``map_fn`` on each element of ``collection``.
        The type of ``collection`` is preserved.
    """
    if collection is None:
        return None
    if isinstance(collection, (tuple, list)):
        return type(collection)(map_fn(x) for x in collection)
    if isinstance(collection, dict):
        return {k: map_fn(v) for k, v in collection.items()}
    return map_fn(collection)


def ensure_tuple(x) -> tuple[Any, ...]:
    """Converts ``x`` into a tuple.

    * If ``x`` is ``None``, then ``tuple()`` is returned.
    * If ``x`` is a tuple, then ``x`` is returned as-is.
    * If ``x`` is a list, then ``tuple(x)`` is returned.
    * If ``x`` is a dict, then ``tuple(v for v in x.values())`` is returned.

    Otherwise, a single element tuple of ``(x,)`` is returned.

    Args:
        x (Any): The input to convert into a tuple.

    Returns:
        tuple: A tuple of ``x``.
    """
    if x is None:
        return ()
    if isinstance(x, (str, bytes, bytearray)):
        return (x,)
    if isinstance(x, collections.abc.Sequence):
        return tuple(x)
    if isinstance(x, dict):
        return tuple(x.values())
    return (x,)


class IteratorFileStream(io.RawIOBase):
    """Class used to convert iterator of bytes into a file-like binary stream object.

    Original implementation found `here <https://stackoverflow.com/questions/6657820/how-to-convert-an-iterable-to-a-stream/20260030#20260030>`_.

    .. note

        A usage example ``f = io.BufferedReader(IteratorFileStream(iterator), buffer_size=buffer_size)``

    Args:
        iterator: An iterator over bytes objects
    """

    def __init__(self, iterator):
        self.leftover = None
        self.iterator = iterator

    def readinto(self, b):
        try:
            l = len(b)  # max bytes to read
            if self.leftover:
                chunk = self.leftover
            else:
                chunk = next(self.iterator)
            output, self.leftover = chunk[:l], chunk[l:]
            b[:len(output)] = output
            return len(output)
        except StopIteration:
            return 0  #EOF

    def readable(self):
        return True


def iterate_with_callback(iterator, total_len, callback=None):
    """Invoke ``callback`` after each chunk is yielded from ``iterator``.

    Args:
        iterator (Iterator): The iterator, which should yield chunks of data.
        total_len (int): The total length of the iterator.
        callback (Callable[[int, int], None], optional): The callback to invoke after
            each chunk of data is yielded back to the caller. Defaults to None, for no callback.

            It is called with the cumulative size of all chunks yielded thus far and the ``total_len``.
    """
    current_len = 0

    if callback is not None:
        # Call the callback for any initialization
        callback(current_len, total_len)

    for chunk in iterator:
        current_len += len(chunk)
        yield chunk
        if callback is not None:
            callback(current_len, total_len)
