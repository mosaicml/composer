# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import abc
import dataclasses
from functools import wraps
import time
from types import TracebackType
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import yahp as hp

from composer.core.callback import Callback
from composer.core.state import State
from composer.core.types import Batch, DataLoader
from composer.datasets.dataloader import WrappedDataLoader
from composer.utils.string_enum import StringEnum


@dataclasses.dataclass
class ProfilerEventHandlerHparams(hp.Hparams, abc.ABC):
    flush_every_n_batches: int = hp.optional("Flush frequency in batches", default=100)
    buffering: int = hp.optional("Python file buffering", default=-1)

    @abc.abstractmethod
    def initialize_object(self) -> ProfilerEventHandler:
        pass


class ProfilerEventHandler(Callback, abc.ABC):

    def process_duration_event(
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

    def process_instant_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        epoch: int,
        step: int,
        wall_clock_time_ns: int,
        perf_counter_time_ns: int,
    ) -> None:
        del name, categories, epoch, step, wall_clock_time_ns, perf_counter_time_ns  # unused
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


class MosaicProfilerAction(StringEnum):
    SKIP = "skip"
    ACTIVE = "active"


class MosaicProfiler:

    def __init__(self, state: State, event_handlers: Sequence[ProfilerEventHandler], active: int, repeat: Optional[int],
                 skip_first_epoch: bool, wait: int) -> None:
        self._names_to_markers: Dict[str, Marker] = {}
        self._event_handlers = event_handlers
        self._state = state
        self._state.train_dataloader = self._wrap_dataloaders_with_markers(self._state.train_dataloader, "train")
        self._state.eval_dataloader = self._wrap_dataloaders_with_markers(self._state.eval_dataloader, "eval")
        self._wait = wait
        self._active = active
        self._repeat = repeat
        self._skip_first_epoch = skip_first_epoch
        self._action = MosaicProfilerAction.SKIP

    def get_action(self):
        if self._state.epoch == 0 and self._skip_first_epoch:
            return MosaicProfilerAction.SKIP
        # do wait, then warump, then active, up to repeat times per cycle
        cycle_len = self._wait + self._active
        if self._repeat is not None and self._state.batch_idx >= cycle_len * self._repeat:
            # exhausted the repeat
            return MosaicProfilerAction.SKIP
        position_in_cycle = self._state.batch_idx % cycle_len
        if position_in_cycle < self._wait:
            return MosaicProfilerAction.SKIP
        return MosaicProfilerAction.ACTIVE

    def marker(self,
               name: str,
               always_record: bool = False,
               categories: Union[List[str], Tuple[str, ...]] = tuple()) -> Marker:
        if name not in self._names_to_markers:
            self._names_to_markers[name] = Marker(self, name, always_record, categories)
        self._names_to_markers[name].categories = categories
        return self._names_to_markers[name]

    def record_duration_event(self, marker: Marker, is_start: bool, wall_clock_time_ns: int, perf_counter_time_ns: int):
        for handler in self._event_handlers:
            handler.process_duration_event(
                name=marker.name,
                categories=marker.categories,
                epoch=self._state.epoch,
                step=self._state.step,
                is_start=is_start,
                wall_clock_time_ns=wall_clock_time_ns,
                perf_counter_time_ns=perf_counter_time_ns,
            )

    def record_instant_event(self, marker: Marker, wall_clock_time_ns: int, perf_counter_time_ns: int):
        for handler in self._event_handlers:
            handler.process_instant_event(name=marker.name,
                                          categories=marker.categories,
                                          epoch=self._state.epoch,
                                          step=self._state.step,
                                          wall_clock_time_ns=wall_clock_time_ns,
                                          perf_counter_time_ns=perf_counter_time_ns)

    def _wrap_dataloaders_with_markers(self, dataloader: DataLoader, name: str) -> DataLoader:
        return ProfiledDataLoader(self, dataloader, name)


class Marker:

    def __init__(self,
                 mosaic_profiler: MosaicProfiler,
                 name: str,
                 always_record: bool,
                 categories: Union[List[str], Tuple[str, ...]] = tuple()) -> None:
        self._instrumentation = mosaic_profiler
        self.name = name
        self.always_record = always_record
        self.categories = categories
        if name in mosaic_profiler._names_to_markers:
            if mosaic_profiler._names_to_markers[name] is not self:
                raise RuntimeError(
                    f"{self.__class__.__name__} should not be instantiated directly. Instead, use {mosaic_profiler.__class__.__name__}.marker(name)"
                )
        self._started = False
        self._action_at_start = None

    def start(self) -> None:
        if self._started:
            raise RuntimeError(f"Attempted to start marker {self.name}; however, this marker is already started")
        self._action_at_start = self._instrumentation.get_action()
        if self._action_at_start == MosaicProfilerAction.ACTIVE or self.always_record:
            self._instrumentation.record_duration_event(
                self,
                is_start=True,
                wall_clock_time_ns=time.time_ns(),
                perf_counter_time_ns=time.perf_counter_ns(),
            )
        self._started = True

    def finish(self) -> None:
        if not self._started:
            raise RuntimeError(f"Attempted to finish marker {self.name}; however, this marker is not yet started")

        if self._action_at_start == MosaicProfilerAction.ACTIVE or self.always_record:
            self._instrumentation.record_duration_event(
                self,
                is_start=False,
                wall_clock_time_ns=time.time_ns(),
                perf_counter_time_ns=time.perf_counter_ns(),
            )
        self._started = False

    def instant(self) -> None:
        if self._instrumentation.get_action() == MosaicProfilerAction.ACTIVE or self.always_record:
            self._instrumentation.record_instant_event(
                self,
                wall_clock_time_ns=time.time_ns(),
                perf_counter_time_ns=time.perf_counter_ns(),
            )

    def __enter__(self) -> Marker:
        self.start()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.finish()

    def __call__(self, func: Optional[Callable] = None) -> Union[Marker, Callable]:
        if func is None:
            return self  # decorator style @Marker()

        @wraps(func)
        def wrapped(*args, **kwargs):
            with self:
                func(*args, **kwargs)

        return wrapped
