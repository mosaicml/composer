from __future__ import annotations

import abc
import time
import dataclasses
from types import TracebackType
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

from composer.core.state import State
from composer.core.types import Batch, DataLoader
from composer.core.callback import Callback
from composer.datasets.dataloader import WrappedDataLoader

import yahp as hp


@dataclasses.dataclass
class ProfilerEventHandlerHparams(hp.Hparams, abc.ABC):
    flush_every_n_batches: int = hp.optional("Flush frequency in batches", default=100)
    buffering: int = hp.optional("Python file buffering", default=-1)

    @abc.abstractmethod
    def initialize_object(self) -> ProfilerEventHandler:
        pass

class ProfilerEventHandler(Callback, abc.ABC):

    def process_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        is_start: bool,
        epoch: int,
        step: int,
        wall_clock_time_ns: int,
        perf_counter_time_ns: int,
    ) -> None:
        del name, categories, is_start, epoch, step, wall_clock_time_ns, perf_counter_time_ns  # unused
        pass


class ProfiledDataLoader(WrappedDataLoader):

    def __init__(self, profiler: MosaicProfiler, dataloader: DataLoader, name: str) -> None:
        super().__init__(dataloader)
        self._mosaic_profiler = profiler
        self._marker = profiler.marker(f"dataloader/{name}", categories=["dataloader"])
        self._iterator: Optional[Iterator[Batch]] = None

    def __iter__(self) -> ProfiledDataLoader:
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self) -> Batch:
        assert self._iterator is not None
        self._marker.start()
        try:
            return next(self._iterator)
        finally:
            self._marker.finish()


class MosaicProfiler:

    def __init__(self, state: State, event_handlers: Sequence[ProfilerEventHandler]) -> None:
        self._names_to_markers: Dict[str, Marker] = {}
        self._event_handlers = event_handlers
        self._state = state
        self._state.train_dataloader = self._wrap_dataloaders_with_markers(self._state.train_dataloader, "train")
        self._state.eval_dataloader = self._wrap_dataloaders_with_markers(self._state.eval_dataloader, "eval")

    def marker(self, name: str, categories: Union[List[str], Tuple[str, ...]] = tuple()) -> Marker:
        if name not in self._names_to_markers:
            self._names_to_markers[name] = Marker(self, name, categories)
        self._names_to_markers[name].categories = categories
        return self._names_to_markers[name]

    def record_event(self, marker: Marker, is_start: bool, wall_clock_time_ns: int, perf_counter_time_ns: int):
        for handler in self._event_handlers:
            handler.process_event(
                name=marker.name,
                categories=marker.categories,
                epoch=self._state.epoch,
                step=self._state.step,
                is_start=is_start,
                wall_clock_time_ns=wall_clock_time_ns,
                perf_counter_time_ns=perf_counter_time_ns,
            )

    def _wrap_dataloaders_with_markers(self, dataloader: DataLoader, name: str) -> DataLoader:
        return ProfiledDataLoader(self, dataloader, name)


class Marker:

    def __init__(self,
                 mosaic_profiler: MosaicProfiler,
                 name: str,
                 categories: Union[List[str], Tuple[str, ...]] = tuple()) -> None:
        self._instrumentation = mosaic_profiler
        self.name = name
        self.categories = categories
        if name in mosaic_profiler._names_to_markers:
            if mosaic_profiler._names_to_markers[name] is not self:
                raise RuntimeError(
                    f"{self.__class__.__name__} should not be instantiated directly. Instead, use {mosaic_profiler.__class__.__name__}.marker(name)"
                )
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError(f"Cannot start an {self.__class__.__name__} that is already started")
        self._instrumentation.record_event(
            self,
            is_start=True,
            wall_clock_time_ns=time.time_ns(),
            perf_counter_time_ns=time.perf_counter_ns(),
        )
        self._started = True

    def finish(self) -> None:
        if not self._started:
            raise RuntimeError(f"Cannot finish a {self.__class__.__name__} that is not started")
        self._instrumentation.record_event(
            self,
            is_start=False,
            wall_clock_time_ns=time.time_ns(),
            perf_counter_time_ns=time.perf_counter_ns(),
        )
        self._started = False

    def __enter__(self) -> Marker:
        self.start()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.finish()
