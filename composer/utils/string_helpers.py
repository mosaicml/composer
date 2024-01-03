# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for string manipulation."""


def partial_format(s, *args, **kwargs):
    """Format a string with a partial set of arguments.

    Since `str.format()` raises a `KeyError` if a format key is missing from the arguments, this
    function allows for a partial set of arguments to be provided. Any missing arguments will be
    left as-is in the string.
    """
    result = s
    done = False
    while not done:
        try:
            result = s.format(*args, **kwargs)
            done = True
        except IndexError as e:  # Missing positional arg
            args += ('{}',)
        except KeyError as e:  # Missing keyword arg
            key = e.args[0]
            kwargs[key] = '{' + key + '}'

    return result
