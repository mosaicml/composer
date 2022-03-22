# Copyright 2021 MosaicML. All Rights Reserved.

"""Outputs profiling data in JSON trace format."""

from __future__ import annotations

import json
import os
import queue
import time
from typing import IO, TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from composer.loggers import LogLevel
from composer.profiler import ProfilerEventHandler
from composer.utils import dist

if TYPE_CHECKING:
    from composer.core.state import State
    from composer.core.time import Timestamp
    from composer.loggers import Logger

__all__ = ["JSONTraceHandler"]


class JSONTraceHandler(ProfilerEventHandler):
    """Records trace events in `JSON trace format <https://\\
    docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview>`_.

    Traces are output to ``output_directory``.  Traces can be visualized using the Chrome Trace Viewer.
    To view in a Google Chrome browser, navigate to ``chrome://tracing`` and load the JSON trace file.

    Args:
        flush_every_n_batches (int): Interval at which to flush the logfile. (Default: ``100`` batches)
        buffering (int, optional): Buffering parameter passed to :meth:`open` when opening the logfile.
            (Default: ``-1`` for the system default)
        filename_format (str, optional): Format string for the filename.

            The following format variables are available:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{run_name}``         | The name of the training run. See                     |
            |                        | :attr:`~composer.core.logging.Logger.run_name`.       |
            +------------------------+-------------------------------------------------------+
            | ``{rank}``             | The global rank, as returned by                       |
            |                        | :func:`~composer.utils.dist.get_global_rank`.         |
            +------------------------+-------------------------------------------------------+
            | ``{local_rank}``       | The local rank of the process, as returned by         |
            |                        | :func:`~composer.utils.dist.get_local_rank`.          |
            +------------------------+-------------------------------------------------------+
            | ``{world_size}``       | The world size, as returned by                        |
            |                        | :func:`~composer.utils.dist.get_world_size`.          |
            +------------------------+-------------------------------------------------------+
            | ``{local_world_size}`` | The local world size, as returned by                  |
            |                        | :func:`~composer.utils.dist.get_local_world_size`.    |
            +------------------------+-------------------------------------------------------+
            | ``{node_rank}``        | The node rank, as returned by                         |
            |                        | :func:`~composer.utils.dist.get_node_rank`.           |
            +------------------------+-------------------------------------------------------+

            .. note::

                When training with multiple devices (i.e. GPUs), ensure that ``'{rank}'`` appears in the format.
                Otherwise, multiple processes may attempt to write to the same file.

            Consider the following example when using default value of '{run_name}/profiler_traces/rank_{rank}.json',
            and the rank of the current process is ``0``.

            >>> json_trace_handler = JSONTraceHandler(filename_format='{run_name}/profiler_traces/rank_{rank}.json')
            >>> trainer = Trainer(..., profiler_trace_handlers=[json_trace_handler], run_name='foo')
            >>> json_trace_handler.filename
            'foo/profiler_traces/rank_0.json'

            Default: ``'{run_name}/profiler_traces/rank_{rank}.json'``

        artifact_name_format (str, optional): Format string for the trace file's artifact name.
        
            The trace file will be periodically logged (according to ``flush_every_n_batches``) as a file artifact.
            The artifact name will be determined by this format string.

            .. seealso:: :meth:`~composer.core.logging.Logger.log_file_artifact` for file artifact logging.

            The same format variables for ``filename_format`` are available. Setting this parameter to ``None``
            (the default) will use the same format string as ``filename_format``. It is sometimes helpful to deviate
            from this default. For example, when ``filename_format`` contains an absolute path, it is recommended to
            set this parameter explicitely, so the absolute path does not appear in any artifact stores.

            Leading slashes (``'/'``) will be stripped.
            
            Default: ``None`` (which uses the same format string as ``filename_format``)
    """

    def __init__(self,
                 flush_every_n_batches: int = 100,
                 buffering: int = -1,
                 filename_format: str = '{run_name}/profiler_traces/rank_{rank}.json',
                 artifact_name_format: Optional[str] = None) -> None:
        self.buffering = buffering
        self.flush_every_n_batches = flush_every_n_batches
        self.filename_format = filename_format
        if artifact_name_format is None:
            artifact_name_format = filename_format.replace(os.path.sep, "/")
        artifact_name_format = artifact_name_format
        self._file: Optional[IO] = None
        self._is_first_line = True
        self._buffer = queue.SimpleQueue()
        self._run_name = None

    @property
    def filename(self) -> str:
        """The filename for the profiler trace."""
        if self._run_name is None:
            raise RuntimeError("The run name is not set. The engine should have been set on Event.INIT")
        name = self.filename_format.format(
            rank=dist.get_global_rank(),
            local_rank=dist.get_local_rank(),
            world_size=dist.get_world_size(),
            local_world_size=dist.get_local_world_size(),
            node_rank=dist.get_node_rank(),
            run_name=self._run_name,
        )

        return name

    @property
    def artifact_name(self) -> str:
        """The artifact name for the profiler trace."""
        if self._run_name is None:
            raise RuntimeError("The run name is not set. The engine should have been set on Event.INIT")
        name = self.filename_format.format(
            rank=dist.get_global_rank(),
            local_rank=dist.get_local_rank(),
            world_size=dist.get_world_size(),
            local_world_size=dist.get_local_world_size(),
            node_rank=dist.get_node_rank(),
            run_name=self._run_name,
        )

        name.lstrip("/")

        return name

    def init(self, state: State, logger: Logger) -> None:
        del state  # unused
        self._run_name = logger.run_name
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self._file = open(self.filename, "x", buffering=self.buffering)
        self._file.write("[\n")
        wall_clock_ns = time.time_ns()
        self._record_event(
            name="process_name",
            ph="M",  # metadata
            wall_clock_ns=wall_clock_ns,
            tid=os.getpid(),
            pid=dist.get_global_rank(),
            args={"name": f"Rank {dist.get_global_rank()} training loop process"})
        self._record_event(
            name="thread_name",
            ph="M",  # metadata
            wall_clock_ns=wall_clock_ns,
            tid=os.getpid(),
            pid=dist.get_global_rank(),
            args={"name": f"Training Loop"})
        self._record_event(
            name="thread_sort_index",
            ph="M",  # metadata
            wall_clock_ns=wall_clock_ns,
            tid=os.getpid(),
            pid=dist.get_global_rank(),
            args={"sort_index": 0})  # training loop thread should be first
        self._record_event(
            name="global_rank",
            ph="M",  # metadata
            wall_clock_ns=wall_clock_ns,
            tid=os.getpid(),
            pid=dist.get_global_rank(),
            args={"value": dist.get_global_rank()})
        self._record_event(
            name="process_sort_index",
            ph="M",  # metadata
            wall_clock_ns=wall_clock_ns,
            tid=os.getpid(),
            pid=dist.get_global_rank(),
            args={"sort_index": dist.get_global_rank()})  # sort index for processes should be the global rank
        # Synchronize the clocks
        # Each rank will record a timestamp at approxmately the same real world time
        clock_sync_a = time.time_ns()
        dist.barrier()  # syncronize all ranks
        clock_sync_time_ns = time.time_ns()
        dist.barrier()  # another barrier to bound the error
        clock_sync_b = time.time_ns()
        clock_sync_error_bound = clock_sync_b - clock_sync_a
        self._record_event(
            name="clock_sync_timestamp_us",
            ph="M",  # metadata
            wall_clock_ns=wall_clock_ns,
            tid=os.getpid(),
            pid=dist.get_global_rank(),
            args={"value": clock_sync_time_ns // 1000})

        self._record_event(
            name="clock_sync_error_bound",
            ph="M",  # metadata
            wall_clock_ns=wall_clock_ns,
            tid=os.getpid(),
            pid=dist.get_global_rank(),
            args={"value": clock_sync_error_bound // 1000})

    def batch_end(self, state: State, logger: Logger) -> None:
        if int(state.timer.batch_in_epoch) % self.flush_every_n_batches == 0:
            self._flush(logger)

    def epoch_end(self, state: State, logger: Logger) -> None:
        del state  # unused
        self._flush(logger)

    def close(self, state: State, logger: Logger):
        del state  # unused
        if self._file is not None:
            self._flush(logger)
            self._file.write("\n]")
            self._file.flush()
            self._file.close()
            self._file = None

    def _flush(self, logger: Logger):
        assert self._file is not None, "flush is only called when the file is open"
        while True:
            try:
                event = self._buffer.get_nowait()
            except queue.Empty:
                break
            entry = json.dumps(event, indent=None)
            if not self._is_first_line:
                self._file.write(",\n")
            self._is_first_line = False
            self._file.write(entry)
        self._file.flush()
        os.fsync(self._file.fileno())
        logger.file_artifact(LogLevel.FIT, artifact_name=self.artifact_name, file_path=self.filename, overwrite=True)

    def process_duration_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        is_start: bool,
        timestamp: Timestamp,
        wall_clock_time_ns: int,
        global_rank: int,
        pid: int,
    ) -> None:
        ph = "B" if is_start else "E"
        args = {}
        args["epoch"] = timestamp.epoch.value
        args["batch"] = timestamp.batch.value
        self._record_event(
            name=name,
            categories=",".join(categories),
            ph=ph,
            wall_clock_ns=wall_clock_time_ns,
            pid=global_rank,
            args=args,
            tid=pid,
        )

    def process_instant_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        timestamp: Timestamp,
        wall_clock_time_ns: int,
        global_rank: int,
        pid: int,
    ) -> None:
        args = {}
        args["epoch"] = timestamp.epoch.value
        args["batch"] = timestamp.batch.value
        self._record_event(
            name=name,
            categories=",".join(categories),
            ph="i",
            wall_clock_ns=wall_clock_time_ns,
            args=args,
            pid=global_rank,
            tid=pid,
            s="p",  # mark instant event for at process level
        )

    def process_counter_event(self, name: str, categories: Union[List[str], Tuple[str, ...]], wall_clock_time_ns: int,
                              global_rank: int, pid: int, values: Dict[str, Union[int, float]]) -> None:
        self._record_event(
            name=name,
            categories=",".join(categories),
            ph='C',  # counter event
            wall_clock_ns=wall_clock_time_ns,
            pid=global_rank,
            tid=pid,
            args=values)

    def _record_event(self, name: str, ph: str, wall_clock_ns: int, pid: int, tid: int, categories: str = "", **kwargs):
        """Helper function to record an event in the trace.

        Args:
            name (str): Event name
            categories (str): Comma-seperated string of event categories
            ph (str): Event type. Should be one of the following
                Duration Events: ``B`` (begin), ``E`` (end)
                Complete Events: ``X``
                Instant Events: ``i``
                Counter Events: ``C``
                Async Events: ``b`` (nestable start), ``n`` (nestable instant), ``e`` (nestable end)
                Flow events: ``s`` (start), ``t`` (step), ``f`` (end)
                Sample events: ``P``
                Object Events ``N`` (created), ``O`` (snapshot), ``D`` (destroyed)
                Metadata Events: ``M``
                Memory Dump Events: ``V`` (global), ``v`` (process)
                Mark Events: ``R``
                Clock Sync Events ``c``
            wall_clock_ns (int): Wall clock time, in nanoseconds.
            tid (int): :meth:`threading.get_ident` value for the event
            pid (int): :meth:`os.get_pid` value for the event
            kwargs: Any extra info to record with the event, such as event specific fields.
        """
        data = {
            "name": name,
            "cat": categories,
            "ph": ph,
            "ts": wall_clock_ns // 1000,  # tracing clock timestamp, in microseconds
            "pid": pid,
            "tid": tid,
            **kwargs,
        }
        self._buffer.put_nowait(data)
