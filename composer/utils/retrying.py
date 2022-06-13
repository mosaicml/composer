# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Retry helper."""

from __future__ import annotations

import collections.abc
import functools
import random
import time
from typing import Any, Callable, Sequence, Type, TypeVar, Union, cast, overload

TCallable = TypeVar('TCallable', bound=Callable)

__all__ = ['retry']


@overload
def retry(
    exc_class: Union[Type[Exception], Sequence[Type[Exception]]] = ...,
    num_attempts: int = ...,
    initial_backoff: float = ...,
    max_jitter: float = ...,
) -> Callable[[TCallable], TCallable]:
    ...


@overload
def retry(exc_class: TCallable) -> TCallable:
    # Use the decorator without parenthesis
    ...


# error: Type "(TCallable@retry) -> TCallable@retry" cannot be assigned to type "(func: Never) -> Never"
def retry(  # type: ignore
    exc_class: Union[TCallable, Type[Exception], Sequence[Type[Exception]]] = Exception,
    num_attempts: int = 3,
    initial_backoff: float = 1.0,
    max_jitter: float = 0.5,
):
    """Decorator to retry a function with backoff and jitter.

    Attempts are spaced out with ``initial_backoff + 2**num_attempts + random.random() * max_jitter`` seconds.

    Example:
    .. testcode::

        from composer.utils import retry

        num_tries = 0

        @retry(RuntimeError, num_attempts=3, initial_backoff=0.1)
        def flaky_function():
            global num_tries
            if num_tries < 2:
                num_tries += 1
                raise RuntimeError("Called too soon!")
            return "Third time's a charm."

        print(flaky_function())

    .. testoutput::

        Third time's a charm.

    Args:
        exc_class (Type[Exception] | Sequence[Type[Exception]]], optional): The exception class or classes to retry.
            Defaults to Exception.
        num_attempts (int, optional): The total number of attempts to make. Defaults to 3.
        initial_backoff (float, optional): The initial backoff, in seconds. Defaults to 1.0.
        max_jitter (float, optional): The maximum amount of random jitter to add. Defaults to 0.5.

            Increasing the ``max_jitter`` can help prevent overloading a resource when multiple processes in parallel
            are calling the same underlying function.
    """
    if num_attempts < 1:
        raise ValueError('num_attempts must be at leats 1')

    def wrapped_func(func: TCallable) -> TCallable:

        @functools.wraps(func)
        def new_func(*args: Any, **kwargs: Any):
            i = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exc_class as e:
                    if i + 1 == num_attempts:
                        raise e
                    else:
                        time.sleep(initial_backoff * 2**i + random.random() * max_jitter)
                        i += 1

        return cast(TCallable, new_func)

    if not isinstance(exc_class, collections.abc.Sequence) and not (isinstance(exc_class, type) and
                                                                    issubclass(exc_class, Exception)):
        # Using the decorator without (), like @retry_with_backoff
        func = cast(TCallable, exc_class)
        exc_class = Exception

        return wrapped_func(func)

    return wrapped_func
