# Copyright 2021 MosaicML. All Rights Reserved.

"""Dynamically import a Python object (e.g. module, class, function, ...)."""

import importlib
import textwrap
from typing import Any

__all__ = ["import_object", "NLP_IMPORT_MESSAGE"]

NLP_IMPORT_MESSAGE = textwrap.dedent("""\
	Composer was installed without NLP support. To use NLP-related packages, with Composer, run `pip install mosaicml[nlp]`
	if using pip or `conda install -c conda-forge datasets transformers` if using Anaconda.""")


def import_object(name: str) -> Any:
    """Dynamically import a Python object (e.g. class, function, ...).

    .. note::

        To dynamically import a module, use :func:`importlib.import_module`.

    Args:
        name (str): The path to the Python object to import.

            Separate the module name and class name with a ``':'`` (e.g. ``'path.to.module:function_name'``).

            Example:

                >>> from composer.utils import import_object
                >>> import_object('functools:partial')
                <class 'functools.partial'>

            .. note::

                The module name must be discoverale with the Python path, as determined by :attr:`sys.path`.

    Returns:
        Any: The imported object.
    """
    module_name, object_name = name.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, object_name)
