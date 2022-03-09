import importlib
from typing import Any
"""Dynamically import a Python object (e.g. module, class, function, ...)."""

__all__ = ["import_object"]


def import_object(name: str) -> Any:
    """Dynamically import a Python object (e.g. class, function, ...).

    .. note::

        To dynamically import a module, use :func:`importlib.import_module`.

    Args:
        name (str): The path to the Python object to import.

            Separate the module name and class name with a ``':'`` (e.g. ``'path.to.module:function_name'``).

            Example:

                >>> dynamic_import('functools:partial')
                <class 'functools.partial'>

            .. note::

                The module name must be discoverale with the Python path, as determined by :attr:`sys.path`.

    Returns:
        Any: The imported object.
    """
    module_name, object_name = name.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, object_name)
