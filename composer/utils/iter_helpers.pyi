import collections.abc
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar, Union, overload

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
def map_collection(singleton: T, map_fn: Callable[[T], V], /) -> V:
    ...

def ensure_tuple(x: Union[T, Tuple[T, ...], List[T], Dict[Any, T]]) -> Tuple[T, ...]:
    ...


def zip_collection(singleton: Any, *others: Any) -> Generator[Tuple[Any, ...], None, None]:
    ...

def iterate_with_pbar(iterator: Iterator[TSized], progress_bar: Optional[tqdm.tqdm] = ...) -> Iterator[TSized]:
    ...
