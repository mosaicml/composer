# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import textwrap
import threading
import time
from typing import TYPE_CHECKING, Dict, cast

from composer.callbacks import memory_monitor
from composer.core.callback import Callback

if TYPE_CHECKING:
    from composer.core.logging.logger import Logger
    from composer.core.state import State


class SystemProfiler(Callback):
    """The SystemProfiler records system level statistics.

    Args:
        profile_cpu (bool): Whether to record cpu statistics (Default: ``True``)
        profile_memory (bool): Whether to record memory statistics (Default: ``False``)
        profile_disk (bool): Whether to record disk I/O statistics (Default: ``False``)
        profile_net (bool): Whether to record network I/O statistics (Default: ``False``)
        stats_thread_interval_seconds (float): Interval to record system-level stats, in seconds. (Default: every ``0.5`` seconds)
    """

    def __init__(self,
                 profile_cpu: bool = True,
                 profile_memory: bool = False,
                 profile_disk: bool = False,
                 profile_net: bool = False,
                 stats_thread_interval_seconds: float = 0.5) -> None:

        self.profile_cpu = profile_cpu
        self.profile_disk = profile_disk
        self.profile_memory = profile_memory
        self.profile_net = profile_net
        self.stats_thread_interval_seconds = stats_thread_interval_seconds

        try:
            # Attempt an import of psutil in init to ensure it is installed
            import psutil
            del psutil
        except ImportError as e:
            raise ImportError(
                "Please install composer with pip install composer[perf] to use the state.profiler") from e

    def init(self, state: State, logger: Logger):
        del logger  # unused
        if state.profiler is None:
            raise RuntimeError(
                textwrap.dedent("""\
                    To use the dataloader state.profiler, state.profiler must be set.
                    Make sure to run composer with the state.profiler -- i.e. with the `--profiler` CLI flag."""))

        # Start the stats thread
        threading.Thread(target=self._stats_thread, daemon=True, args=[state]).start()

    def _stats_thread(self, state: State):
        import psutil  # already checked that it's installed in init
        psutil.disk_io_counters.cache_clear()
        psutil.net_io_counters.cache_clear()
        if self.profile_cpu:
            psutil.cpu_percent()  # spin it once to clear the default 0.0 value on the first call

        while True:
            if self.profile_cpu:
                cpu_percent = psutil.cpu_percent()
                state.profiler.marker(name="cpu", state=state, categories=["cpu"]).counter({"cpu_percent": cpu_percent})

            if self.profile_memory:
                cuda_memory_stats = memory_monitor.get_memory_report()
                for name, val in cuda_memory_stats.items():
                    state.profiler.marker(f"memory/cuda/{name}", state=state,
                                          categories=["memory"]).counter({name: val})
                swap_memory = psutil.swap_memory()
                state.profiler.marker("memory/swap", state=state, categories=["memory"]).counter({
                    "used_gb": swap_memory.used / 2**9,
                    "free_gb": swap_memory.free / 2**9
                })
                virtual_memory = psutil.virtual_memory()
                state.profiler.marker("memory/virtual", state=state, categories=["memory"]).counter({
                    "used_gb": virtual_memory.used / 2**9,
                    "available_gb": virtual_memory.available / 2**9
                })

            if self.profile_disk:
                disk_io_counters = cast(Dict[str, psutil._common.sdiskio], psutil.disk_io_counters(perdisk=True))
                for disk_name, disk_stats in disk_io_counters.items():
                    for field_name in ("read_count", "write_count", "read_bytes", "write_bytes", "read_time",
                                       "write_time", "busy_time"):
                        state.profiler.marker(f"disk/{disk_name}/{field_name}", state=state,
                                              categories=["disk"
                                                         ]).counter({"field_name": getattr(disk_stats, field_name)})

            if self.profile_net:
                net_io_counters = cast(Dict[str, psutil._common.snetio], psutil.net_io_counters(pernic=True))
                for nic, nic_stats in net_io_counters.items():
                    state.profiler.marker(f"network/{nic}/kb_sent", state=state,
                                          categories=["net"]).counter({"kb_sent": nic_stats.bytes_sent / 2**3})
                    state.profiler.marker(f"network/{nic}/kb_recv", state=state,
                                          categories=["net"]).counter({"kb_recv": nic_stats.bytes_recv / 2**3})

            time.sleep(self.stats_thread_interval_seconds)
