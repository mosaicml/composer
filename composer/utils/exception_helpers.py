"""Helpers for working with exceptions."""

import contextlib
import sys
from types import TracebackType
from typing import Callable, Type

__all__ = ["print_traceback_before_cleanup"]

# Keep track of the last processed exception, so it is not printed twice.
# This heuristic will work most of the time, but will fail when all of the following conditions are met:
# 1. The same exception object is raised twice in a row.
# 2. For the second time the exception is raised, it is not guarded by `print_traceback_before_cleanup`.
# 3. The caller catches and surpresses the exception (e.g. via a try/except).
# 4. No other exceptions are raised in the meantime.
# Meeting these four conditions is very unlikely to happen (Exception objects are rarely, if ever, re-used,
# and even if so, it is unlikely that one of the codepaths would use `print_traceback_before_cleanup` while
# the other one wouldn't.
_last_exc = None

# Store a reference to the original excepthook
# `sys.__excepthook__` is not necessarily ``sys.excepthook`` if something else modified the excepthook
_original_excepthook = sys.excepthook


# Override the `sys.excepthook` to not print any tracebacks in `printed_tracebacks`
def _new_excepthook(exc_type: Type[BaseException], exc_value: BaseException, exc_tb: TracebackType):
    global _last_exc
    if exc_value is _last_exc:
        return
    # The same exception would not be passed through the excepthook twice, unless if was actually raised again,
    # so _last_exc should be cleared.
    _last_exc = None
    return _original_excepthook(exc_type, exc_value, exc_tb)


sys.excepthook = _new_excepthook


@contextlib.contextmanager
def print_traceback_before_cleanup(cleanup: Callable[[], None]):
    """Print any traceback before invoking ``cleanup``.

    This helper function returns a context manager. It ensures that any exception traceback
    is printed *before* ``cleanup`` is invoked, and also ensures that the traceback is not printed twice
    (if it remains uncaught. However, any exception will be still propagated back up the stack.)

    .. note::

        Unlike the default behavior, :func:`sys.excepthook` will be invoked for all exceptions raised within the context
        manager, not just exceptions that remain uncaught by the calling code.

    Example usage:

    .. testsetup::

        from composer.utils import print_traceback_before_cleanup

    .. testcode::

        with print_traceback_before_cleanup(cleanup=lambda: print("Hello, world!")):
            raise RuntimeError("Exception")

    .. testoutput::

        Traceback (most recent call last):
        ...
        RuntimeError: Exception
        Hello, world!

    Args:
        cleanup (Callable[[], None]): Function to invoke after the exception was printed.
    """
    try:
        yield
    except:
        exc_type, exc_value, exc_tb = sys.exc_info()
        assert exc_type is not None, "There was an exception!"
        assert exc_value is not None, "There was an exception!"
        # Clear _last_exc, since any exception here is a new exception
        # and should be printed again.
        global _last_exc
        _last_exc = None
        try:
            # Print the traceback (and call any exception handlers)
            sys.excepthook(exc_type, exc_value, exc_tb)
            pass
        finally:
            # Ensure that the traceback is not processed again, since it was already processed via
            # sys.excepthook above
            _last_exc = exc_value
            raise
    finally:
        cleanup()
