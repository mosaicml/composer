import collections.abc
import io
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union, overload

T = TypeVar("T")
V = TypeVar("V")
KT = TypeVar("KT")

TSized = TypeVar("TSized", bound=collections.abc.Sized)



@overload
def map_collection(tuple_of_elements: Tuple[T, ...], map_fn: Callable[[T], V], /) -> Tuple[V, ...]:
    ...


@overload
def map_collection(list_of_elements: List[T], map_fn: Callable[[T], V], /) -> List[V]:
    ...


@overload
def map_collection(dict_of_elements: Dict[KT, T], map_fn: Callable[[T], V], /) -> Dict[KT, V]:
    ...


@overload
def map_collection(none: None, map_fn: Callable[[Any], Any], /) -> None:
    ...


@overload
def map_collection(singleton: T, map_fn: Callable[[T], V], /) -> V:
    ...


def ensure_tuple(union_of_all_types: Union[T, Sequence[T], Dict[Any, T], None]) -> Tuple[T, ...]:
    ...


class IteratorFileStream(io.RawIOBase):
    def __init__(self, iterator):
        ...

    def readinto(self, b) -> int:
        ...

    def readable(self) -> bool:
        ...

class IteratorWithCallback(Generic[TSized]):
    def __init__(
        self,
        iterator: Iterator[TSized],
        total_len: int,
        callback: Optional[Callable[[int, int], None]] = ...,
    ):
        ...

    def __next__(self) -> TSized:
        ...

    def __iter__(self) -> IteratorWithCallback[TSized]:
        ...
