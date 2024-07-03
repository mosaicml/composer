import collections.abc
import io
from typing import Any, Callable, Iterator, Optional, Sequence, TypeVar, Union, overload

T = TypeVar("T")
V = TypeVar("V")
KT = TypeVar("KT")

TSized = TypeVar("TSized", bound=collections.abc.Sized)



@overload
def map_collection(tuple_of_elements: tuple[T, ...], map_fn: Callable[[T], V], /) -> tuple[V, ...]:
    ...


@overload
def map_collection(list_of_elements: list[T], map_fn: Callable[[T], V], /) -> list[V]:
    ...


@overload
def map_collection(dict_of_elements: dict[KT, T], map_fn: Callable[[T], V], /) -> dict[KT, V]:
    ...


@overload
def map_collection(none: None, map_fn: Callable[[Any], Any], /) -> None:
    ...


@overload
def map_collection(singleton: T, map_fn: Callable[[T], V], /) -> V:
    ...


def ensure_tuple(union_of_all_types: Union[T, Sequence[T], dict[Any, T], None]) -> tuple[T, ...]:
    ...


class IteratorFileStream(io.RawIOBase):
    def __init__(self, iterator):
        ...

    def readinto(self, b) -> int:
        ...

    def readable(self) -> bool:
        ...

def iterate_with_callback(
    iterator: Iterator[TSized],
    total_len: int,
    callback: Optional[Callable[[int, int], None]] = ...,
):
    ...
