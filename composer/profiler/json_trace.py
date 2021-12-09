# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import dataclasses
import json
import os
import queue
import threading
import time
from typing import IO, Dict, List, Optional, Tuple, Union, cast

import yahp as hp

import composer.callbacks.memory_monitor as memory_monitor
from composer.core.profiler import ProfilerEventHandler, ProfilerEventHandlerHparams
from composer.core.state import State
from composer.core.types import Logger
from composer.utils import ddp
from composer.utils.run_directory import get_relative_to_run_directory


@dataclasses.dataclass
class JSONTraceHandlerHparams(ProfilerEventHandlerHparams):
    """Parameters for the :class:`JSONTraceHandler`."""
    flush_every_n_batches: int = hp.optional("Interval at which to flush the logfile.", default=100)
    buffering: int = hp.optional("Buffering parameter passed to :meth:`open` when opening the logfile.", default=-1)
    output_directory: str = hp.optional("Directory, relative to the run directory, to store traces.",
                                        default="mosaic_profiler")
    profile_cpu: bool = hp.optional("Whether to record cpu statistics", default=True)
    profile_memory: bool = hp.optional("Whether to record memory statistics", default=False)
    profile_disk: bool = hp.optional("Whether to record disk statistics", default=False)
    profile_net: bool = hp.optional("Whether to record network statistics", default=False)
    stats_thread_interval_seconds: float = hp.optional("Interval to record stats, in seconds.", default=0.5)

    def initialize_object(self) -> JSONTraceHandler:
        return JSONTraceHandler(**dataclasses.asdict(self))


class JSONTraceHandler(ProfilerEventHandler):
    """Records trace events in `JSON trace format <https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview>`_.

    Args:
        flush_every_n_batches (int): Interval at which to flush the logfile. (Default: ``100`` batches)
        buffering (int, optional): Buffering parameter passed to :meth:`open` when opening the logfile.
            (Default: ``-1`` for the system default)
        output_directory (str): Directory, relative to the run directory, to store traces.
            Each trace will be called ``rank_XXX.trace.json`` within this directory, and have an associated metadata file
            called ``rank_XXX.metadata.trace.json``, where ``XXX`` is the global rank.
            (Default: ``mosaic_profiler`` within the run directory)
        profile_cpu (bool): Whether to record cpu statistics (Default: ``True``)
        profile_memory (bool): Whether to record memory statistics (Default: ``False``)
        profile_disk (bool): Whether to record disk I/O statistics (Default: ``False``)
        profile_net (bool): Whether to record network I/O statistics (Default: ``False``)
        stats_thread_interval_seconds (float): Interval to record system-level stats, in seconds. (Default: every ``0.5`` seconds)
    """

    def __init__(self,
                 flush_every_n_batches: int = 100,
                 buffering: int = -1,
                 output_directory: str = "mosaic_profiler",
                 profile_cpu: bool = True,
                 profile_memory: bool = False,
                 profile_disk: bool = False,
                 profile_net: bool = False,
                 stats_thread_interval_seconds: float = 0.5) -> None:
        self.buffering = buffering
        self.profile_disk = profile_disk
        self.profile_memory = profile_memory
        self.profile_net = profile_net
        self.profile_cpu = profile_cpu
        self.flush_every_n_batches = flush_every_n_batches
        self.output_directory = output_directory
        self.stats_thread_interval_seconds = stats_thread_interval_seconds
        self._file: Optional[IO] = None
        self._is_first_line = True
        self._buffer = queue.SimpleQueue()
        self._ddp_rank = ddp.get_global_rank()
        try:
            # Attempt an import of psutil in init to ensure it is installed
            import psutil
            del psutil
        except ImportError as e:
            raise ImportError("Please install composer with pip install composer[perf] to use the profiler") from e

    def init(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        os.makedirs(get_relative_to_run_directory(self.output_directory), exist_ok=True)
        self._file = open(get_relative_to_run_directory(
            os.path.join(self.output_directory, f"rank_{ddp.get_global_rank()}.trace.json")),
                          "x",
                          buffering=self.buffering)
        self._file.write("[\n")
        wall_clock_ns = time.time_ns()
        self._record_event(
            name="process_name",
            ph="M",  # metadata
            wall_clock_ns=wall_clock_ns,
            tid=threading.get_ident(),
            pid=os.getpid(),
            args={"name": f"Rank {ddp.get_global_rank()} training loop process"})
        self._record_event(
            name="thread_name",
            ph="M",  # metadata
            wall_clock_ns=wall_clock_ns,
            tid=threading.get_ident(),
            pid=os.getpid(),
            args={"name": f"Training Loop"})
        self._record_event(
            name="global_rank",
            ph="M",  # metadata
            wall_clock_ns=wall_clock_ns,
            tid=threading.get_ident(),
            pid=os.getpid(),
            args={"value": ddp.get_global_rank()})
        # Syncronize the clocks
        # Each rank will record a timestamp at approxmately the same real world time
        clock_sync_a = time.time_ns()
        ddp.barrier()  # syncronize all ranks
        clock_sync_time_ns = time.time_ns()
        ddp.barrier()  # another barrier to bound the error
        clock_sync_b = time.time_ns()
        clock_sync_error_bound = clock_sync_b - clock_sync_a

        # Write the static metadata to a trace file in object format
        metadata_filename = f"rank_{ddp.get_global_rank()}.metadata.trace.json"
        metadata_file = get_relative_to_run_directory(os.path.join(self.output_directory, metadata_filename))
        with open(metadata_file, "x") as f:
            json.dump(
                {
                    "clock_sync_timestamp_us": clock_sync_time_ns // 1000,
                    "clock_sync_error_us": clock_sync_error_bound // 1000,
                    "displayTimeUnit": "ns",
                    "global_rank": ddp.get_global_rank(),
                    "traceEvents": [],
                }, f)

        # Start the stats thread
        threading.Thread(target=self._stats_thread, daemon=True).start()

    def batch_end(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if (state.batch_idx + 1) % self.flush_every_n_batches == 0:
            self._flush()

    def training_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._flush()

    def epoch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._flush()

    def close(self):
        if self._file is not None:
            self._flush()
            self._file.write("\n]")
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
            if not self._is_first_line:
                self._file.write(",\n")
            self._is_first_line = False
            self._file.write(entry)

    def process_duration_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        is_start: bool,
        epoch: Optional[int],
        step: Optional[int],
        wall_clock_time_ns: int,
        process_id: int,
        thread_id: int,
    ) -> None:
        ph = "B" if is_start else "E"
        args = {}
        if epoch is not None:
            args["epoch"] = epoch
        if step is not None:
            args["step"] = step
        self._record_event(
            name=name,
            categories=",".join(categories),
            ph=ph,
            wall_clock_ns=wall_clock_time_ns,
            pid=process_id,
            args=args,
            tid=thread_id,
        )

    def process_instant_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        epoch: Optional[int],
        step: Optional[int],
        wall_clock_time_ns: int,
        process_id: int,
        thread_id: int,
    ) -> None:
        args = {}
        if epoch is not None:
            args["epoch"] = epoch
        if step is not None:
            args["step"] = step
        self._record_event(
            name=name,
            categories=",".join(categories),
            ph="i",
            wall_clock_ns=wall_clock_time_ns,
            args=args,
            pid=process_id,
            tid=thread_id,
            s="p",  # mark instant event for at process level
        )

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

    def _stats_thread(self):
        import psutil  # already checked that it's installed in init
        psutil.disk_io_counters.cache_clear()
        psutil.net_io_counters.cache_clear()
        if self.profile_cpu:
            psutil.cpu_percent()  # spin it once to clear the default 0.0 value on the first call
        while True:
            wall_clock_time_ns = time.time_ns()
            cuda_memory_stats = memory_monitor.get_memory_report()
            disk_io_counters = cast(Dict[str, psutil._common.sdiskio], psutil.disk_io_counters(perdisk=True))
            net_io_counters = cast(Dict[str, psutil._common.snetio], psutil.net_io_counters(pernic=True))
            cpu_percent = psutil.cpu_percent()
            if self.profile_cpu:
                self._record_event(
                    name=f"cpu",
                    categories="cpu",
                    ph='C',  # counter event
                    wall_clock_ns=wall_clock_time_ns,
                    pid=os.getpid(),
                    tid=threading.get_ident(),
                    args={"cpu_percent": cpu_percent})
            if self.profile_memory:
                for name, val in cuda_memory_stats.items():
                    self._record_event(
                        name=f"cuda_memory/{name}",
                        categories="memory",
                        ph='C',  # counter event
                        wall_clock_ns=wall_clock_time_ns,
                        pid=os.getpid(),
                        tid=threading.get_ident(),
                        args={name: val})
                swap_memory = psutil.swap_memory()
                self._record_event(
                    name=f"memory/swap",
                    categories="memory",
                    ph='C',  # counter event
                    wall_clock_ns=wall_clock_time_ns,
                    pid=os.getpid(),
                    tid=threading.get_ident(),
                    args={
                        "used_gb": swap_memory.used / 2**9,
                        "free_gb": swap_memory.free / 2**9
                    })
                virtual_memory = psutil.virtual_memory()
                self._record_event(
                    name=f"memory/virtual",
                    categories="memory",
                    ph='C',  # counter event
                    wall_clock_ns=wall_clock_time_ns,
                    pid=os.getpid(),
                    tid=threading.get_ident(),
                    args={
                        "used_gb": virtual_memory.used / 2**9,
                        "available_gb": virtual_memory.available / 2**9
                    })
            if self.profile_disk:
                for disk_name, disk_stats in disk_io_counters.items():
                    for field_name in ("read_count", "write_count", "read_bytes", "write_bytes", "read_time",
                                       "write_time", "busy_time"):
                        self._record_event(
                            name=f"disk/{disk_name}/{field_name}",
                            categories=f"disk,disk/{disk_name},disk/{field_name}",
                            ph='C',  # counter event
                            wall_clock_ns=wall_clock_time_ns,
                            pid=os.getpid(),
                            tid=threading.get_ident(),
                            args={"field_name": getattr(disk_stats, field_name)})
            if self.profile_net:
                for nic, nic_stats in net_io_counters.items():
                    self._record_event(
                        name=f"network/{nic}/kb_sent",
                        categories=f"network,network/{nic},network/kb_sent",
                        ph='C',  # counter event
                        wall_clock_ns=wall_clock_time_ns,
                        pid=os.getpid(),
                        tid=threading.get_ident(),
                        args={"kb_sent": nic_stats.bytes_sent / 2**3})
                    self._record_event(
                        name=f"network/{nic}/kb_recv",
                        categories=f"network,network/{nic},network/kb_recv",
                        ph='C',  # counter event
                        wall_clock_ns=wall_clock_time_ns,
                        pid=os.getpid(),
                        tid=threading.get_ident(),
                        args={"kb_recv": nic_stats.bytes_recv / 2**3})
            time.sleep(self.stats_thread_interval_seconds)
