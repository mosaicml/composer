from __future__ import annotations
import abc

from typing import Dict, Generator, Optional, Sequence, Type, List, Tuple, Union
from types import TracebackType

import time

from composer.core.state import State
from composer.core.types import Batch, DataLoader

class ProfilerEventHandler(abc.ABC):
    def process_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str]],
        is_start: bool,
        epoch: int,
        step: int,
        wall_clock_time_ns: int,
        perf_counter_time_ns: int,
    ) -> None:
        pass

class MosaicProfiler:
    def __init__(self, state: State, event_handlers: Sequence[ProfilerEventHandler]) -> None:
        self._names_to_markers: Dict[str, Marker] = {}
        self._event_handlers = event_handlers
        self._state = state
        state.da

    def marker(self, name: str, categories: Union[List[str], Tuple[str]] = tuple()) -> Marker:
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



def _inject_before_dataloader_event(self, dataloader: DataLoader) -> Generator[Batch, None, None]:
    dataloader_iterator = iter(dataloader)
    while True:
        self.engine.run_event('BEFORE_DATALOADER_FETCH')
        try:
            batch = next(dataloader_iterator)
            yield batch
        finally:
            self.engine.run_event('AFTER_DATALOADER_FETCH')
        
class Marker:

    def __init__(self, mosaic_profiler: MosaicProfiler, name: str, categories: Union[List[str], Tuple[str]] = tuple()) -> None:
        self._instrumentation = mosaic_profiler
        self.name = name
        self.categories = categories
        if name in mosaic_profiler._names_to_markers:
            if mosaic_profiler._names_to_markers[name] is not self:
                raise RuntimeError(f"{self.__class__.__name__} should not be instantiated directly. Instead, use {mosaic_profiler.__class__.__name__}.marker(name)")
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
    
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.finish()
