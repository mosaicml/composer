# Copyright 2021 MosaicML. All Rights Reserved.

"""Outputs profiling data in JSON trace format."""

from __future__ import annotations

import json
import os
import pathlib
import queue
import time
import gzip
from typing import Dict, List, Optional, Tuple, Union

from collections import OrderedDict

from composer.core.state import State
from composer.core.time import Timestamp
from composer.loggers import Logger, LogLevel
from composer.profiler._profiler_action import ProfilerAction
from composer.profiler._trace_handler import TraceHandler
from composer.profiler.json_trace_merger import merge_traces
from composer.utils import dist, ensure_folder_is_empty, format_name_with_dist, format_name_with_dist_and_time

__all__ = ["JSONTraceHandler"]


class JSONTraceHandler(TraceHandler):
    """Records trace events in `JSON trace format <https://\\
    docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview>`_.

    Traces are output to ``output_directory``.  Traces can be visualized using the Chrome Trace Viewer.
    To view in a Google Chrome browser, navigate to ``chrome://tracing`` and load the JSON trace file.

    Args:
        folder_format (str, optional): Format string for the folder containing the trace files.

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

            Consider the following example when using default value of '{run_name}/traces',
            and the rank of the current process is ``0``.

            >>> json_trace_handler = JSONTraceHandler(folder_format='{run_name}/traces')
            >>> trainer = Trainer(..., profiler_trace_handlers=[json_trace_handler], run_name='foo')
            >>> json_trace_handler.trace_folder
            'foo/traces'

            Default: ``'{run_name}/traces'``

        filename_format (str, optional): A format string describing how to name trace files.
            (default: ``'ep{epoch}-ba{batch}-rank{rank}.json'``)

            At the end of each batch where :meth:`~composer.profiler.Profiler.get_action` returns
            :attr:`~composer.profiler._profiler_action.ProfilerAction.ACTIVE_AND_SAVE`, trace files are saved
            approximately to ``{trace_folder}/{filename_format.format(...)}``.

            The following format variables are available:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{run_name}``         | The name of the training run. See                     |
            |                        | :attr:`~composer.core.logging.Logger.run_name`.       |
            +------------------------+-------------------------------------------------------+
            | ``{rank}``             | The global rank, as returned by                       |
            |                        | :func:`~.dist.get_global_rank`.                       |
            +------------------------+-------------------------------------------------------+
            | ``{local_rank}``       | The local rank of the process, as returned by         |
            |                        | :func:`~.dist.get_local_rank`.                        |
            +------------------------+-------------------------------------------------------+
            | ``{world_size}``       | The world size, as returned by                        |
            |                        | :func:`~.dist.get_world_size`.                        |
            +------------------------+-------------------------------------------------------+
            | ``{local_world_size}`` | The local world size, as returned by                  |
            |                        | :func:`~.dist.get_local_world_size`.                  |
            +------------------------+-------------------------------------------------------+
            | ``{node_rank}``        | The node rank, as returned by                         |
            |                        | :func:`~.dist.get_node_rank`.                         |
            +------------------------+-------------------------------------------------------+
            | ``{epoch}``            | The total epoch count, as returned by                 |
            |                        | :meth:`~composer.core.time.Timer.epoch`.              |
            +------------------------+-------------------------------------------------------+
            | ``{batch}``            | The total batch count, as returned by                 |
            |                        | :meth:`~composer.core.time.Timer.batch`.              |
            +------------------------+-------------------------------------------------------+
            | ``{batch_in_epoch}``   | The batch count in the current epoch, as returned by  |
            |                        | :meth:`~composer.core.time.Timer.batch_in_epoch`.     |
            +------------------------+-------------------------------------------------------+
            | ``{sample}``           | The total sample count, as returned by                |
            |                        | :meth:`~composer.core.time.Timer.sample`.             |
            +------------------------+-------------------------------------------------------+
            | ``{sample_in_epoch}``  | The sample count in the current epoch, as returned by |
            |                        | :meth:`~composer.core.time.Timer.sample_in_epoch`.    |
            +------------------------+-------------------------------------------------------+
            | ``{token}``            | The total token count, as returned by                 |
            |                        | :meth:`~composer.core.time.Timer.token`.              |
            +------------------------+-------------------------------------------------------+
            | ``{token_in_epoch}``   | The token count in the current epoch, as returned by  |
            |                        | :meth:`~composer.core.time.Timer.token_in_epoch`.     |
            +------------------------+-------------------------------------------------------+

            Consider the following scenario, where:

            *   The :attr:`~.Logger.run_name` is 'awesome-training-run'
            *   The default ``trace_folder_format='{run_name}/traces'`` is used.
            *   The default ``name_format='ep{epoch}-ba{batch}-rank{rank}.json'`` is used.
            *   The current epoch count is ``1``.
            *   The current batch count is ``42``.

            Each rank (process) will save traces to::

                awesome-training-run/traces/ep1-ba42-rank0.json
                awesome-training-run/traces/ep1-ba42-rank1.json
                awesome-training-run/traces/ep1-ba42-rank2.json
                ...

        artifact_name_format (str, optional): Format string for the trace file's artifact name.
            (default: ``'{run_name}/traces/ep{epoch}-ba{batch}-rank{rank}.json'``)
        
            Whenever a trace file is saved, it is also logged as a file artifact according to this format string.
            The same format variables as for ``filename_format`` are available.

            .. seealso:: :meth:`~composer.core.logging.Logger.file_artifact` for file artifact logging.

            Leading slashes (``'/'``) will be stripped.

        merged_trace_filename_format (str, optional): Format string for the merged trace filename.
            (default: ``'node{node_rank}.json'``)

            Each rank writes a separate trace file at the end of each profiling cycle. However, when visualizing
            traces, it is generally helpful to merge traces together into a single file. This allows the traces
            across all ranks to be shown in a single view. To 

            The same format variables as for ``filename_format`` are available. The merged trace file is saved
            approximately to ``{trace_folder}/{merged_trace_filename_format.format(...)}`` on the local rank zero
            process for each node.

            If specified (the default), the local rank zero process merges together all traces files from that node,
            across all profiling cycles, into a single trace file. The merged trace file is written to the filename
            specified by this format string. There will be one merged trace file per node.

            To disable merging, set this parameter to ``None``.

            .. warning::

                Trace merging blocks the training loop. When profiling live training runs, it is recommended to
                disable trace merging by setting this parameter to ``None``. Instead, traces should be merged together
                in post-processing steps. See :mod:`composer.profiler.json_trace_merger` for additional info.
        
        merged_trace_artifact_name_format (str, optional): Format string for the merged trace file's artifact name.
            (default: ``'{run_name}/traces/merged_trace.json'``)

            The same format variables as for ``filename_format`` are available.

            This parameter has no effect if ``merged_trace_filename_format`` is None.

        overwrite (bool, optional): Whether to overwrite existing traces. (default: ``False``)
            If ``False``, the :meth:`trace_folder` (as determined by the ``trace_folder_format`` argument)
            must be empty when training starts.
        
        buffering (int, optional): Buffering parameter passed to :meth:`open` when opening the trace file.
            (Default: ``-1`` for the system default)

        num_trace_cycles_to_keep (int, optional): 
    """

    def __init__(
        self,
        folder_format: str = '{run_name}/traces',
        filename_format: str = 'ep{epoch}-ba{batch}-rank{rank}.json',
        artifact_name_format: str = '{run_name}/traces/ep{epoch}-ba{batch}-rank{rank}.json',
        merged_trace_filename_format: Optional[str] = 'merged_trace.json',
        merged_trace_artifact_name_format: str = '{run_name}/traces/merged_trace.json',
        *,
        overwrite: bool = False,
        buffering: int = -1,
        num_trace_cycles_to_keep: int = -1,
    ):
        self.buffering = buffering
        self.folder_format = folder_format
        self.overwrite = overwrite
        self.filename_format = filename_format
        self.artifact_name_format = artifact_name_format
        self.merged_trace_filename_format = merged_trace_filename_format
        self.merged_trace_artifact_name_format = merged_trace_artifact_name_format
        self.saved_traces: Dict[Timestamp, List[str]] = OrderedDict()
        self.num_trace_cycles_to_keep = num_trace_cycles_to_keep

        self._queue: queue.Queue[str] = queue.Queue()
        self._is_trace_active = False

    def init(self, state: State, logger: Logger) -> None:
        del state  # unused
        trace_folder = format_name_with_dist(self.folder_format, run_name=logger.run_name)

        os.makedirs(trace_folder, exist_ok=True)
        if not self.overwrite:
            ensure_folder_is_empty(trace_folder)
        # Ensure all ranks checked that the folder is empty before proceeding
        dist.barrier()

    def batch_start(self, state: State, logger: Logger) -> None:
        if state.profiler is None:
            raise RuntimeError(("The Composer Profiler was not enabled, which is required to use the "
                                f"{type(self).__name__}. To enable, set the `prof_schedule` argument of the Trainer."))
        if state.profiler.get_action(state) != ProfilerAction.SKIP and not self._is_trace_active:
            # Starting a new profiling cycle
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

            self._is_trace_active = True

    def batch_end(self, state: State, logger: Logger) -> None:
        assert state.profiler is not None
        timestamp = state.timer.get_timestamp()
        trace_folder = format_name_with_dist(self.folder_format, run_name=logger.run_name)
        if state.profiler.get_action(state) == ProfilerAction.SKIP and self._is_trace_active:
            # no longer active, but was previously active.
            # Epty the queue and save the trace file
            trace_filename = os.path.join(trace_folder,)
            with open(trace_filename, 'w+') as f:
                is_first_line = True
                f.write('[\n')
                while True:
                    try:
                        s = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if not is_first_line:
                        s = ",\n" + s
                    is_first_line = False
                    f.write(s)
                f.write('\n]\n')
            artifact_name = format_name_with_dist_and_time(self.artifact_name_format, logger.run_name, timestamp)
            logger.file_artifact(LogLevel.BATCH,
                                 artifact_name=artifact_name,
                                 file_path=trace_filename,
                                 overwrite=self.overwrite)
            # Gather the filenames
            self.saved_traces[timestamp] = dist.all_gather_object(trace_filename)

            # Ensure that all traces have been saved.
            dist.barrier()

            if self.merged_trace_filename_format is not None and dist.get_local_rank() == 0:
                # Merge together all traces from the node into one file
                start_rank = dist.get_global_rank()
                end_rank = dist.get_global_rank() + dist.get_local_world_size()
                trace_files_to_merge = self.saved_traces[timestamp][start_rank:end_rank]
                merged_trace_filename = os.path.join(
                    trace_folder,
                    format_name_with_dist_and_time(self.merged_trace_filename_format, logger.run_name, timestamp))
                merge_traces(merged_trace_filename, *trace_files_to_merge)
                merged_trace_artifact_name = format_name_with_dist_and_time(self.merged_trace_artifact_name_format,
                                                                            logger.run_name, timestamp)
                logger.file_artifact(LogLevel.BATCH,
                                     artifact_name=merged_trace_artifact_name,
                                     file_path=merged_trace_artifact_name,
                                     overwrite=True)

            self._is_trace_active = False

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
        entry = json.dumps(data, indent=None)
        self._queue.put_nowait(entry)

    def process_chrome_json_trace_file(self, filepath: pathlib.Path) -> None:
        with (gzip.open(filepath, 'rt') if str(filepath).endswith('.gz') else open(filepath, "r")) as f:
            # It may be an incomplete trace file that is missing the closing ] bracket, as is permitted
            # in the chrome json format spec
            trace_data_str = f.read().strip()
            if trace_data_str.startswith('[') and not trace_data_str.endswith(']'):
                trace_data_str += ']'
            trace_data = json.loads(trace_data_str)

        if isinstance(trace_data, dict):
            event_list = trace_data["traceEvents"]
        else:
            event_list = trace_data

        if not isinstance(event_list, list):
            raise TypeError("A trace file should either be a dict or a list")

        for entry in event_list:
            entry['pid'] = dist.get_global_rank()  # override the PID to the global rank
            entry_s = json.dumps(entry, indent=None)
            self._queue.put_nowait(entry_s)
