import collections.abc
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union, overload

import tqdm

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


@overload
def ensure_tuple(x: None) -> Tuple:
    ...


@overload
def ensure_tuple(x: str) -> Tuple[str]:
    ...


@overload
def ensure_tuple(x: bytearray) -> Tuple[bytearray]:
    ...


@overload
def ensure_tuple(x: bytes) -> Tuple[bytes]:
    ...


@overload
def ensure_tuple(x: Tuple[T, ...]) -> Tuple[T, ...]:
    ...


@overload
def ensure_tuple(x: List[T]) -> Tuple[T, ...]:
    ...


@overload
def ensure_tuple(x: range) -> Tuple:
    ...


@overload
def ensure_tuple(x: Sequence[T]) -> Tuple[T, ...]:
    ...


@overload
def ensure_tuple(x: Dict[Any, T]) -> Tuple[T, ...]:
    ...


@overload
def ensure_tuple(x: Union[T, None, Sequence[T], Dict[Any, T]]) -> Tuple[T, ...]:
    ...


def iterate_with_pbar(iterator: Iterator[TSized], progress_bar: Optional[tqdm.tqdm] = ...) -> Iterator[TSized]:
    ...
