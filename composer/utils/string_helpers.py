# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for string manipulation."""


def partial_format(s, *args, **kwargs):
    """Format a string with a partial set of arguments.

    Since `str.format()` raises a `KeyError` if a format key is missing from the arguments, this
    function allows for a partial set of arguments to be provided.

    For example:

    >>> partial_format('{foo} {bar}', foo='Hello')
    'Hello {bar}'

    >>> partial_format('{foo} {bar}', foo='Hello', bar='World')
    'Hello World'
    """
    result = s
    done = False
    while not done:
        try:
            result = s.format(*args, **kwargs)
            done = True
        except KeyError as e:
            key = e.args[0]
            kwargs[key] = '{' + key + '}'

    return result
