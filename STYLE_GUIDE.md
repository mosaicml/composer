# Composer Style Guide

This guide serves as a rule-of-thumb for how code should be organized and formatted in Composer.
For general python styling questions not covered in this document, consult Google's
[Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## 1. Code Formatting

Composer uses the [yapf](https://github.com/google/yapf) formatter for general formatting,
[isort](https://github.com/PyCQA/isort) to sort imports,
[docformatter](https://github.com/myint/docformatter) to format docstrings, and
[pylint](https://github.com/PyCQA/pylint) as the general style checker.

To (auto)format code, run

```bash
yapf -rip .  # for yapf, auto-formats in place
isort .  # for isort, auto-formats in place

# for docformatter, auto-formats in place
docformatter -ri --wrap-summaries 120 --wrap-descriptions 120 composer tests examples

pylint composer tests examples  # for pylint, does NOT auto-format

```

The configuration is stored in [pyproject.toml](pyproject.toml).

All modified files must have their pylint errors addressed
(eventually, the entire codebase will become pylint compliant).
Please do NOT address pylint errors in unrelated, unmodified files,
as that will clutter the PR.


## 2. Type Annotations and Typechecking

Composer aims to annotate all functions with type annotations (introduced in
[PEP 526](https://www.python.org/dev/peps/pep-0526/). Type annotations help statically catch `TypeError` and
`AttributeError` bugs, in addition to other benefits, as outlined in the PEP.

Composer uses [pyright](https://github.com/microsoft/pyright)
to validate type annotations. To check typing, run:

```bash
pyright .
```

Our pyright configuration is stored in [pyproject.toml](pyproject.toml).

While pyright warnings will not cause a build to fail, they should be reviewed as they can identify potential bugs.

### Debugging

Here are some suggestions to deal with pyright errors:

1. Suppose a variable could be one of multiple types, like the following:

    ```python
    def foo(x: Union[int, None]):
        """
        Args:
            x (int or None): Foo parameter
        """
        return x + 5  # type error -- None + 5 is not allowed!
    ```

    Pyright will complain since `None + 5` is not a valid operation.
    Instead, add a check to ensure that `x is not None`:

    ```python
    def foo(x: Union[int, None]):
        if x is None:
            raise TypeError("x must be an integer, not None!")
        return x + 5  # valid
    ```

    Assert statements also work. However, assert statements should NOT be used for data validation
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


## 3. Public APIs
A public API, generally speaking, can be invoked by a user without a leading underscore in any portion of the path.
The following are examples of public APIs:

* Standalone functions in public modules (e.g. `composer.utils.dist.get_world_size`)
* Classes in public modules (e.g. `composer.trainer.checkpoint.CheckpointLoader`)
* Public methods in public classes (e.g. `composer.trainer.checkpoint.CheckpointLoader.load_checkpoint`)

The following rules apply to public APIs:
1. All public APIs must have a docstring
2. All parameters must have type annotations
3. Parameters should be flattened, rather than nested. While this violates the software engineering practice of
    being DRY (don't repeat yourself), it simplifies the user API by making all parameters visible.
4. To minimize user imports, parameters should support native PyTorch types or Python primitives whenever possible.

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
    (or no element). For example, `Tensors = Union[Tensor, Tuple[Tensor, ...], List[Tensor]]`.

    The `composer.utils.ensure_tuple` helper method can convert a singleton, list, or tuple into a tuple.
    For example

    ```python
    def foo(x: Optional[Tensors]) -> Tuple[Tensor, ...]:
        return ensure_tuple(x)  # ensures that the result is always a (potentially empty) tuple of tensors
    ```


## 4. Use of `assert`

`assert` should be used only in test cases and for verifying invariants (likely required for type checking),
not for data validation. As asserts can be disabled in python by using the `-O` flag (e.g. `python -O path/to/script.py`),
they are not guaranteed to run. For data validation, instead use a style like the following:

```python
if parameter is None:
    raise ValueError("parameter must be specified and cannot be None")
```


## 5. Imports and `__init__.py`

All imports in composer should be absolute -- that is, they do not begin with a period.
In addition, all imports should be added into an appropriate section of [setup.py](setup.py).

### 5.1 Optional Dependencies

If an import is not in `install_requires`, then this import MUST be conditionally imported in the code
-- e.g. like this:

```python
def unet():
    try:
        import monai
    except ImportError as e:
        raise ImportError(textwrap.dedent("""\
            Composer was installed without unet support. To use unet with Composer, run: `pip install mosaicml
            [unet]` if using pip or `conda install -c conda-forge monai` if using Anaconda""") from e
```

This style allows users to install composer without the extra dependencies. Otherwise, if the import is global
(i.e. at the top of the file), they would receive ImportErrors when the module is not found.

### 5.2 `__init__.py`

All public classes and functions should be added to the module's `__init__.py`.

```python
from composer.path.to.module.file import MyClass as MyClass
from composer.path.to.module.file import my_func as my_func
```

If a file only contains public functions, then the following is also acceptable:

```python
from composer.path.to.module import my_file as my_file
```


## 6. Hparams and Config Classes

Each class that could be initialized by the hparams will need its configuration variables defined in
[yahp](https://github.com/mosaicml/yahp) dataclass.

Guidelines:
* The fields in the dataclass should roughly follow the init signature of the class being constructed
* `initialize_object(self)` should take any parameters needed to construct the class. It acceptable to
take complex types or other hparam dataclasses, as required, to initialize the object.
