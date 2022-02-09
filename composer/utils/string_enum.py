# Copyright 2021 MosaicML. All Rights Reserved.

"""Base class for Enums containing string values."""
from __future__ import annotations

import textwrap
import warnings
from enum import Enum

__all__ = ["StringEnum"]


class StringEnum(Enum):
    """Base class for Enums containing string values.

    :class:`StringEnum` enforces that all keys are uppercase, and all values are lowercase.
    
    In addition, it offers the following features:

    * ``StringEnum(value)`` will perform a case-insensitive match on both the keys and value,
        and is a no-op if given an existing instance of the class.

        >>> class MyStringEnum(StringEnum):
        ...     KEY = "value"
        >>> MyStringEnum("KeY")  # case-insensitive match on the key
        something
        >>> MyStringEnum("VaLuE")  # case-insensitive match on the value
        something
        >>> MyStringEnum(MyStringEnum.KEY)  # no-op if given an existing instance
        something
    
    * Equality checks support case-insensitive comparisions against strings:

        >>> class MyStringEnum(StringEnum):
        ...     KEY = "value"
        >>> MyStringEnum.KEY == "KeY"  # case-insensitive match on the key
        True
        >>> MyStringEnum.KEY == "VaLuE"  # case-insensitive match on the value
        True
        >>> MyStringEnum.KEY == "something else"
        False
    """
    __hash__ = Enum.__hash__

    def __eq__(self, o: object) -> bool:
        if isinstance(o, str):
            cls_name = self.__class__.__name__
            warnings.warn(
                f"Detected comparision between a string and {cls_name}. Please use {cls_name}({o}) "
                f"to convert both types to {cls_name} before comparing",
                category=UserWarning)
            o_enum = type(self)(o)
            return super().__eq__(o_enum)
        return super().__eq__(o)

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
                raise ValueError(f"Value {value} not found in {cls.__name__}")
        raise TypeError(f"Unable to convert value({value}) of type {type(value)} into {cls.__name__}")
