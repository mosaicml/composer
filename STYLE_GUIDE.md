# Style and Conventions

### 2.1 Style Guide

Composer generally follows Google's
[Python Style Guide](https://google.github.io/styleguide/pyguide.html) for how to format and structure code.

### 2.2. Code Formatting

Composer uses the [yapf](https://github.com/google/yapf) formatter for general formatting,
[isort](https://github.com/PyCQA/isort) to sort imports, and
[docformatter](https://github.com/myint/docformatter) to format docstrings.

To (auto)format code, run

```bash
make style
```

To verify that styling and typing (see below) is correct, run:
```bash
make lint
```

The configuration is stored in [pyproject.toml](pyproject.toml).


## 3. Type Annotations and Typechecking

Composer aims to annotate all functions with type annotations (introduced in
[PEP 526](https://www.python.org/dev/peps/pep-0526/). Type annotations help statically catch `TypeError` and
`AttributeError` bugs, in addition to other benefits, as outlined in the PEP.

Composer uses [pyright](https://github.com/microsoft/pyright)
to validate type annotations. To check typing, run:

```bash
make lint
```

The pyright configuration is stored in [pyproject.toml](pyproject.toml).


### Debugging

Here are some suggestions to deal with pyright errors:

1. Suppose a variable could be one of multiple types, like the following:

    ```python
    def foo(x: Union[int, None]):
        return x + 5  # type error -- None + 5 is not allowed!
    ```

    PyRight will complain since `None + 5` is not a valid operation.
    Instead, add a check to ensure that `x is not None`:

    ```python
    def foo(x: Union[int, None]):
        if x is None:
            raise TypeError("x must be an integer, not None!")
        return x + 5  # valid
    ```

    Assert statements also work. However, assert statements should not be used for data validation
    (see the assert statement section below).
    ```python
    def foo(x: Union[int, None]):
        assert x is not None, "x should never be None"
        return x + 5  # valid
    ```

1. For variables where it is impossible for pyright to infer the correct type, use
[cast](https://docs.python.org/3/library/typing.html#typing.cast).
1. As a last resort, add a `# type: ignore` comment to the line where pyright emits an error.
Immediately following this statement, paste in the error emitted by pyright,
so other contributors will know why this error was silenced.


## 4. Public APIs
A public API, generally speaking, can be invoked by a user without a leading underscore in any portion of the path.
The following are examples of public APIs:

* Standalone functions in public modules (e.g. `composer.utils.dist.get_world_size`)
* Classes in public modules (e.g. `composer.trainer.trainer.Trainer`)
* Public methods in public classes (e.g. `composer.trainer.trainer.Trainer.fit`)
* Public modules (e.g. `composer.trainer.trainer`)

The following rules apply to public APIs:
1. All public APIs must have a docstring (see the Documentation section below)
1. All parameters must have type annotations.
1. To minimize user imports, parameters should should use native PyTorch or Python types whenever possible.

    It is acceptable to use a union of types, so long as one of the options is a primitive. For example, in the
    constructor for `composer.trainer.trainer.Trainer`, the `device` parameter is annotated like the following:

    ```python
    from typing import Optional, Union

    from composer.trainer.devices import Device

    class Trainer:
        def __init__(
            self,
            ...,
            device: Union[str, Device],

        ):
            if isinstance(device, str):
                device = Device(device)
            ...
    ```

    This signature allows a user to pass a string for a device,
    rather than having to import our custom device class.

    Parameters that are for power users (such as `load_object_store`) in the Trainer are exempt from this rule.
    These parameters can require custom imports.

1. Parameters that could take a sequence of elements should also allow `None` or a singleton.
    This simplifies the user API by not having to construct a list (or tuple) to hold a single element
    (or no element). For example, use `Optional[Union[torch.Tensor, Sequence[torch.Tensor]]`.

    The `composer.utils.ensure_tuple` helper method can convert a singleton, list, or tuple into a tuple.
    For example

    ```python
    def foo(x: Optional[Union[Tensor, Sequence[Tensor]]) -> Tuple[Tensor, ...]:
        return ensure_tuple(x)  # ensures that the result is always a (potentially empty) tuple of tensors
    ```


## 5. Use of `assert`

`assert` should be used only in test cases and for verifying invariants (likely required for type checking),
not for data validation. As asserts can be disabled in python by using the `-O` flag (e.g. `python -O path/to/script.py`),
they are not guaranteed to run. For data validation, instead use a style like the following:

```python
if parameter is None:
    raise ValueError("parameter must be specified and cannot be None")
```


## 6. Imports and `__init__.py`

All imports in composer should be absolute -- that is, they do not begin with a period.

### 6.1 External Dependencies
1.  All external dependencies must be specified in both [setup.py](setup.py) for pip and [meta.yaml](meta.yaml)
    for Anaconda.

1.  If a dependency is not core to Composer (e.g. it is for a model, dataset, algorithm, or some callbacks):
    1.  It must be specified in a entry of the `extra_deps` dictionary of [setup.py](setup.py).
        This dictionary groups dependencies that can be conditionally installed. An entry named `foo`
        can be installed with `pip install mosaicml[foo]`. For example, running `pip install mosaicml[unet]`
        will install everything in `install_requires`, along with `monai` and `scikit-learn`.
    1.  It must also be specified in the `run_constrained` and the `test.requires` section.
    1.  The import must be conditionally imported in the code. For example:

        ```python
        def unet():
            try:
                import monai
            except ImportError as e:
                raise ImportError(textwrap.dedent("""\
                    Composer was installed without unet support. To use unet with Composer, run: `pip install mosaicml
                    [unet]` if using pip or `conda install -c conda-forge monai` if using Anaconda""") from e
        ```

        This style allows users to perform minimal install of Composer without triggering `ImportError`s if
        an optional dependency is missing.

    1.  If the dependency is core to Composer, add the dependency to the `install_requires` section of
        [setup.py](./setup.py) and the `requirements.run` section of [meta.yaml](./meta.yaml).

### 6.2 Use of `__all__`

All public modules must define `__all__` to be the list of members that should be re-exported.
The variable is necessary to 1) limit what `from XXX import *` imports, and 2) ensure that the documentation only includes exported members, not unrelated re-imports.

For example, from [composer/callbacks/memory_monitor.py](composer/callbacks/memory_monitor.py)

```python
"""Log memory usage during training."""
import logging
from typing import Dict, Union

import torch.cuda

from composer.core import Logger, State
from composer.core.callback import Callback

log = logging.getLogger(__name__)

__all__ = ["MemoryMonitor"]  # export only the MemoryMonitor, not other imports like `Logger`, `State`, or `Callback`


class MemoryMonitor(Callback):
    ...
```


### 6.3 `__init__.py`

All public classes and functions should be added to the module's `__init__.py`.

```python
from composer.path.to.module.file import MyClass as MyClass
from composer.path.to.module.file import my_func as my_func
```

If a file only contains public functions, then the following is also acceptable:

```python
from composer.path.to.module import my_file as my_file
```


## 7. Documentation

Composer uses [Google Style Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

The following guidelines apply to documentation.
1.  Each function that needs a docstring must have its input arguments and return statement (if not None) annotated.
1.  Each argument annotation should include the type. If the argument has a default value, the type annotation should
    specify "optional", and the docstring should say the default value. Some examples:

    ```python

    def foo(bar: int):
        """Foo.

        Args:
            bar (int): Required bar.
        """
        ...

    def foo2(bar: int = 42):
        """Foo2.

        Args:
            bar (int, optional): The first Argument. Default: ``42``.
        """
        ...

    def foo3(bar: Optional[int] = None):
        """Foo3.

        Args:
            bar (int, optional): The first Argument. Default: ``None``.
        """
        ...

    def foo4(bar: Union[int, str] = 42):
        """Foo4.

        Args:
            bar (int | str, optional): The first Argument. Default: ``42``.
        """
        ...

    def foo5(bar: int) -> int:
        """Foo5.

        Args:
            bar (int): Required bar.

        Returns:
            int: Description of return statement.
        """
        ...

    def foo6(bar: int) -> Tuple[int, str]:
        """Foo6.

        Args:
            bar (int): Required bar.

        Returns:
            a (int): Returned value.
            b (str): Returned value.
        """
        ...
    ```

1.  For examples in docstrings, use `.. doctest::` or
    `.. testcode::` . See the [Sphinx Doctest Extension](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html)
    for all of the available directives. Do not use `.. code-block::` for Python examples, as they are untested.

    Any test fixtures for doctests should go in [docs/source/doctest_fixtures.py](docs/source//doctest_fixtures.py)
    or in a `.. testsetup::` block.

    For example:
    ```python
    def my_function(x: Optional[torch.Tensor]) -> torch.Tensor:
        """blah function

        Args:
            input (torch.Tensor): Your guess.

        Returns:
            torch.Tensor: How good your input is.

        Raises:
            ValueError: If your input is negative.

        Example:
            .. testsetup::

                # optional setup section, not shown in docs
                import torch
                x = torch.randn(42)


            .. testcode::

                # shown in docs; runs after testsetup
                my_function(x)
        """
        ...
    ```

    To check doctests, run:

    ```bash
    cd docs && make clean && make html && make doctest
    ```
