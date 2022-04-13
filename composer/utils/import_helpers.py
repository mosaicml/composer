# Copyright 2021 MosaicML. All Rights Reserved.

"""Dynamically import a Python object (e.g. module, class, function, ...)."""

import importlib
import textwrap
from typing import Any, Optional

__all__ = ["import_object", "MissingConditionalImportError"]


class MissingConditionalImportError(ImportError):

    def __init__(self, extra_deps_group: str, conda_package: str, conda_channel: Optional[str] = 'conda-forge'):
        """Handles errors for external packages that might not be installed.

        Args:
            extra_deps_group (str): the pip package group, found in setup.py. For example, nlp for `mosaicml[nlp]`.
            conda_package (str): the conda packages to install.
            conda_channel (Optional[str]): the conda channel to install packages from.

        Raises: ImportError
        """
        super().__init__(
            textwrap.dedent(f"""\
            Composer was installed without {extra_deps_group} support. To use {extra_deps_group} related packages, with Composer, run `pip install mosaicml[{extra_deps_group}]`
            if using pip or `conda install -c {conda_channel} {conda_package} if using Anaconda."""))


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
