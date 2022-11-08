# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Base class for Enums containing string values."""
from __future__ import annotations

import textwrap
import warnings
from enum import Enum


class StringEnum(Enum):
    """Base class for Enums containing string values.

    This class enforces that all keys are uppercase and all values are lowercase. It also offers
    the following convenience features:

    *   ``StringEnum(value)`` will perform a case-insensitive match on both the keys and value,
        and is a no-op if given an existing instance of the class.

        .. testsetup::

            import warnings

            warnings.filterwarnings(action="ignore", message="Detected comparision between a string")

        .. doctest::

            >>> from composer.utils import StringEnum
            >>> class MyStringEnum(StringEnum):
            ...     KEY = "value"
            >>> MyStringEnum("KeY")  # case-insensitive match on the key
            <MyStringEnum.KEY: 'value'>
            >>> MyStringEnum("VaLuE")  # case-insensitive match on the value
            <MyStringEnum.KEY: 'value'>
            >>> MyStringEnum(MyStringEnum.KEY)  # no-op if given an existing instance
            <MyStringEnum.KEY: 'value'>

        .. testcleanup::

            warnings.resetwarnings()

    *   Equality checks support case-insensitive comparisions against strings:

        .. testsetup::

            import warnings

            warnings.filterwarnings(action="ignore", message="Detected comparision between a string")

        .. doctest::

            >>> from composer.utils import StringEnum
            >>> class MyStringEnum(StringEnum):
            ...     KEY = "value"
            >>> MyStringEnum.KEY == "KeY"  # case-insensitive match on the key
            True
            >>> MyStringEnum.KEY == "VaLuE"  # case-insensitive match on the value
            True
            >>> MyStringEnum.KEY == "something else"
            False

        .. testcleanup::

            warnings.resetwarnings()
    """
    __hash__ = Enum.__hash__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            cls_name = self.__class__.__name__
            warnings.warn(
                f"Detected comparision between a string and {cls_name}. Please use {cls_name}('{other}') "
                f'to convert both types to {cls_name} before comparing.',
                category=UserWarning)
            try:
                o_enum = type(self)(other)
            except ValueError:  # `other` is not a valid enum option
                return NotImplemented
            return super().__eq__(o_enum)
        return super().__eq__(other)

    def __init__(self, *args: object) -> None:
        if self.name.upper() != self.name:
            raise ValueError(
                textwrap.dedent(f"""\
                {self.__class__.__name__}.{self.name} is invalid.
                All keys in {self.__class__.__name__} must be uppercase.
                To fix, rename to '{self.name.upper()}'."""))
        if self.value.lower() != self.value:
            raise ValueError(
                textwrap.dedent(f"""\
                The value for {self.__class__.__name__}.{self.name}={self.value} is invalid.
                All values in {self.__class__.__name__} must be lowercase. "
                To fix, rename to '{self.value.lower()}'."""))

    @classmethod
    def _missing_(cls, value: object) -> StringEnum:
        # Override _missing_ so both lowercase and uppercase names are supported,
        # as well as passing an instance through
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                if value.lower() != value:
                    return cls(value.lower())
                raise ValueError(f'Value {value} not found in {cls.__name__}')
        raise TypeError(f'Unable to convert value({value}) of type {type(value)} into {cls.__name__}')
