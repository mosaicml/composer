# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for string manipulation."""


def partial_format(s, *args, **kwargs) -> str:
    """Format a string with a partial set of arguments.

    Since `str.format()` raises a `KeyError` if a format key is missing from the arguments, this
    function allows for a partial set of arguments to be provided. Any missing arguments will be
    left as-is in the string.
    """
    max_iters = 10_000  # Just in case we get stuck in a loop somehow.
    while max_iters:
        try:
            return s.format(*args, **kwargs)
        except IndexError as e:  # Missing positional arg
            args += ('{}',)
        except KeyError as e:  # Missing keyword arg
            key = e.args[0]
            kwargs[key] = '{' + key + '}'
