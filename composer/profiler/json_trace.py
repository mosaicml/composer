# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import atexit
import dataclasses
import json
import os
import queue
import threading
import time
from typing import IO, List, Optional, Tuple, Union

import yahp as hp

import composer.callbacks.memory_monitor as memory_monitor
from composer.core.event import Event
from composer.core.profiler import ProfilerEventHandler, ProfilerEventHandlerHparams
from composer.core.state import State
from composer.core.types import Logger
from composer.utils.ddp import get_global_rank
from composer.utils.run_directory import get_relative_to_run_directory


@dataclasses.dataclass
class JSONTraceHparams(ProfilerEventHandlerHparams):
    flush_every_n_batches: int = hp.optional("Flush frequency in batches", default=100)
    buffering: int = hp.optional("Python file buffering", default=-1)
    memory_monitor_interval_seconds: float = hp.optional("memory monitor interval", default=0.5)

    def initialize_object(self) -> JSONTrace:
        return JSONTrace(**dataclasses.asdict(self))


class JSONTrace(ProfilerEventHandler):

    def __init__(self, flush_every_n_batches: int, buffering: int, memory_monitor_interval_seconds: float) -> None:
        self._file: Optional[IO] = None
        self._buffering = buffering
        self._flush_every_n_batches = flush_every_n_batches
        self._memory_monitor_interval_seconds = memory_monitor_interval_seconds
        self._buffer = queue.SimpleQueue()

    def _run_event(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.INIT:
            os.makedirs(get_relative_to_run_directory("mosaic_profiler"), exist_ok=True)
            self._file = open(get_relative_to_run_directory(
                os.path.join("mosaic_profiler", f"rank_{get_global_rank()}.trace.json")),
                              "x",
                              buffering=self._buffering)
            self._file.write("[\n")
            threading.Thread(target=self.memory_monitor_thread, daemon=True).start()
            atexit.register(self._close_logfile)
        if event == Event.BATCH_START:
            self._record_step(state)
        if event == Event.BATCH_END:
            if (state.batch_idx + 1) % self._flush_every_n_batches == 0:
                self._flush()
        if event == Event.TRAINING_END:
            self._flush()
        if event == Event.EPOCH_END:
            self._flush()

    def _record_step(self, state: State) -> None:
        wall_clock_time_ns = time.time_ns()
        perf_counter_time_ns = time.perf_counter_ns()
        self._buffer.put_nowait({
            "name": "step",
            "ph": 'C',  # counter event
            "ts": wall_clock_time_ns // 1000,  # tracing clock timestamp, in microseconds
            "tts": perf_counter_time_ns // 1000,  # thread clock timestamp, in microseconds
            "pid": get_global_rank(),
            "tid":
                0,  # right now, all events are thread 0 for the process. But we may want to break this out (e.g. for dataloader workers)
            "args": {
                "step": state.step,
            }
        })
        self._buffer.put_nowait({
            "name": "epoch",
            "ph": 'C',  # counter event
            "ts": wall_clock_time_ns // 1000,  # tracing clock timestamp, in microseconds
            "tts": perf_counter_time_ns // 1000,  # thread clock timestamp, in microseconds
            "pid": get_global_rank(),
            "tid":
                0,  # right now, all events are thread 0 for the process. But we may want to break this out (e.g. for dataloader workers)
            "args": {
                "epoch": state.epoch,
            }
        })

    def _close_logfile(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def _flush(self):
        assert self._file is not None, "flush is only called when the file is open"
        while True:
            try:
                event = self._buffer.get_nowait()
            except queue.Empty:
                break
            entry = json.dumps(event, indent=None)
            self._file.write(entry)
            self._file.write(",\n")

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
        ph = "B" if is_start else "E"
        self._buffer.put_nowait({
            "name": f"{name}",
            "cat": ",".join(categories),
            "ph": ph,
            "ts": wall_clock_time_ns // 1000,  # tracing clock timestamp, in microseconds
            "tts": perf_counter_time_ns // 1000,  # thread clock timestamp, in microseconds
            "pid": get_global_rank(),
            "tid":
                0,  # right now, all events are thread 0 for the process. But we may want to break this out (e.g. for dataloader workers)
            "args": {
                "epoch": epoch,
                "step": step,
            }
        })

    def process_instant_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        epoch: int,
        step: int,
        wall_clock_time_ns: int,
        perf_counter_time_ns: int,
    ) -> None:
        self._buffer.put_nowait({
            "name": f"{name}",
            "cat": ",".join(categories),
            "ph": "i",
            "ts": wall_clock_time_ns // 1000,  # tracing clock timestamp, in microseconds
            "tts": perf_counter_time_ns // 1000,  # thread clock timestamp, in microseconds
            "pid": get_global_rank(),
            "tid":
                0,  # right now, all events are thread 0 for the process. But we may want to break this out (e.g. for dataloader workers)
            "s": "p",  # mark instant event for at process level
            "args": {
                "epoch": epoch,
                "step": step,
            }
        })

    def memory_monitor_thread(self):
        while True:
            wall_clock_time_ns = time.time_ns()
            perf_counter_time_ns = time.perf_counter_ns()
            memory_stats = memory_monitor.get_memory_report()
            for name, val in memory_stats.items():
                self._buffer.put_nowait({
                    "name": f"memory/{name}",
                    "cat": "gpu",
                    "ph": 'C',  # counter event
                    "ts": wall_clock_time_ns // 1000,  # tracing clock timestamp, in microseconds
                    "tts": perf_counter_time_ns // 1000,  # thread clock timestamp, in microseconds
                    "pid": get_global_rank(),
                    "tid":
                        0,  # right now, all events are thread 0 for the process. But we may want to break this out (e.g. for dataloader workers)
                    "args": {
                        name: val,
                    }
                })
            time.sleep(self._memory_monitor_interval_seconds)
