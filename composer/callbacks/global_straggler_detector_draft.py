#from megatron_core.megatron.core.utils import StragglerDetector
#from megatron_core.megatron.core.utils import *
#from mosaicml.composer.megatron import *
from megatron.core.utils import *

from composer.core import Callback, State, Event, Time
from composer.loggers import Logger
from composer.utils import dist
from dataclasses import dataclass
import time
from composer.models.base import ComposerModel
import os


import logging
import math
import operator
import queue
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from types import TracebackType
from typing import List, Optional, Tuple, Type, Union, Any, Callable, Deque, Dict


import torch


log = logging.getLogger(__name__)

__all__ = ["GlobalStragglerDetector"]

class _ValueWithRank:
    """This is an internal class, not for use outside this module
    Attributes:
        _rank (int): rank for the value
        _value (float) : the value it stores, eg elapsed time
        _unit (str) : unit for the value
    """

    def __init__(self, value: float, rank: int, unit: str = "") -> None:
        """Initializer
        Args:
            _value (float): the initial value with which it is inited
            _rank (int): the rank number
            _unit (str) : the unit of the value, eg ms or flops
        """
        self._rank = rank
        self._value = value
        self._unit = unit

    def __lt__(self, other) -> bool:
        """ Check if value of self is smaller than other's value
        Args:
            other (_ValueWithRank): The other object to compare with
        Returns:
            bool: True if lhs._value of operand is less than rhs._value, else False
        """
        return self._value < other._value

    def __gt__(self, other) -> bool:
        """Check if value of self is larger than other's value
        Args:
            other (_ValueWithRank): The other object to compare with
        Returns:
            bool: True if lhs._value of operand is greater than rhs._value, else False
        """
        return self._value > other._value

    def __call__(self) -> Tuple[float, int, str]:
        """Returns the value, the rank, and unit as a Tuple
            
        Returns:
            Tuple[float, int, str]: value, rank, unit
        """
        return self._value, self._rank, self._unit

    def __str__(self) -> str:
        """String representation of the object
        Returns:
            str: strigified object
        """

        return f"{self._value:.2f}{self._unit}/{self._rank}"


@dataclass
class _StragglerData:
    """This is an internal dataclass, not for use outside this module
    Attributes:
        min_elapsed (_ValueWithRank) min iteration time across all ranks
        max_elapsed (_ValueWithRank) max iteration time across all ranks
        min_btime (_ValueWithRank) min cpu time across all ranks
        max_btime (_ValueWithRank) max cpu time across all ranks
        min_temp (_ValueWithRank): min gpu temp across all ranks
        max_temp (_ValueWithRank): max gpu temp across all ranks
        min_power (_ValueWithRank) min gpu power across all ranks
        max_power (_ValueWithRank) max gpu power across all ranks
        min_util (_ValueWithRank): min gpu util across all ranks
        max_util (_ValueWithRank): max gpu util across all ranks
        min_clock (_ValueWithRank): min gpu clock across all ranks
        max_clock (_ValueWithRank) max gpu clock across all ranks
        aflops (List[_ValueWithRank]): sorted array of (_ValueWithRank)
    """

    # gemm time
    min_elapsed = _ValueWithRank(sys.float_info.max, 0, "ms")
    max_elapsed = _ValueWithRank(sys.float_info.min, 0, "ms")
    # get_batch time
    min_btime = _ValueWithRank(sys.float_info.max, 0, "us")
    max_btime = _ValueWithRank(sys.float_info.min, 0, "us")
    # temp
    min_temp = _ValueWithRank(sys.float_info.max, 0, "C")
    max_temp = _ValueWithRank(sys.float_info.min, 0, "C")
    # power
    min_power = _ValueWithRank(sys.float_info.max, 0, "W")
    max_power = _ValueWithRank(sys.float_info.min, 0, "W")
    # util
    min_util = _ValueWithRank(sys.float_info.max, 0, "%")
    max_util = _ValueWithRank(sys.float_info.min, 0, "%")
    # clock
    min_clock = _ValueWithRank(sys.float_info.max, 0, "MHz")
    max_clock = _ValueWithRank(sys.float_info.min, 0, "MHz")
    aflops: List[_ValueWithRank] = None


class StragglerDetector:
    """Singleton Class implementing per rank Straggler Detector
    It use cuda events to time operation of choice using the
    start and stop methods which can be directly invoked using
    the class instance or can be used like a python context.
    After collection, a report() method is available to display
    the collected metrics. It is only supported if CUDA is
    available. megatron/core/README_STRAGGLER.md for more info
    Note:
        The instance and class attributes mentioned below are all
        private to the class and has no use outside the class
    Attributes:
        _off (bool): current state of the toggle
        start (FunctionType): start method
        stop (FunctionType): stop method
        world (int): world size
        rank (int): rank for this instance
        mmcnt (int): number of ranks to report
        port (int): control port
        amp (float): amplification factor for TFLOPs, default 3.0
        toggle (bool): whether to start/stop detector collection
        bdata (bool): when true, just collect get_batch
        dev (int): cuda device
        idx (int): index into the list below
        idx_q (LifoQueue): queue of index
        evt_q (LifoQueue): cuda event queue
        start_events (list[torch.cuda.Event]): cuda start event
        stop_events (list[torch.cuda.Event]): cuda stop event
        start_time (list[int]): start time (wallclock)
        stop_time (list[int]): stop time (wallclock)
        start_batch (list[int]): start time for get_batch
        stop_batch (list[int]): stop time for get_batch
        sock (socket): the controller socket
        ctrlr (Thread): the controller thread
        logger (Logger): the logger instance for this instance
    """

    _configured = False
    """Indicates if the singleton instance is configured or not
    """

    def __new__(cls: Type["StragglerDetector"]) -> "StragglerDetector":
        """Constructor
        Creates an instance of the class if not created
        Args:
            cls (Type[&#39;StragglerDetector&#39;]): The class type
        Returns:
            StragglerDetector: the class instance
        """

        if not hasattr(cls, "_instance"):
            cls._instance = super(StragglerDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initializer
        The inital state of the StragglerDetector instance is disabled.
        The enabled state is indicated using self._off member variable
        and the proerty enabled.
        """
        self._off = True
        self.start = self.null_method
        self.stop = self.null_method
        self.world = 0
        self.rank = 0
        self.mmcnt = 1
        self.port = 0
        self.amp = 3.0
        self.amp = 1.0
        self.toggle = False
        self.bdata = False
        self.dev = None
        self.idx = 0
        self.idx_q = None
        self.evt_q = None
        self.start_events = None
        self.stop_events = None
        self.start_time = None
        self.stop_time = None
        self.start_batch = None
        self.stop_batch = None
        self.sock = None
        self.ctrlr = None
        self.logger = logging.getLogger(__name__)

    def configure(
        self,
        world: int,
        rank: int,
        mmcnt: int = 1,
        amp: float = 3.0,
        port: int = 65535,
        prefill: int = 1024,
        enabled: bool = False,
    ) -> None:
        """This method is called to configure the Singleton instance
        It should be called once per instantiation per process.
        Note:
            The constructor keeps the state of instance disabled
            i.e no collection will happen even when start/stop methods are
            called. Only when enabled is True (self._off is True), the
            start/stop method pointers get assigned the real collection
            methods, otherwise they are initialized with null_method
        Args:
            world (int): World Size
            rank (int): The rank of this trainer
            mmcnt (int, optional): Number of ranks to print for showing Min/Max Etpt.
                                   Defaults to 1.
            amp (float, optional): Set to 3.0 if we only use timers in fwd pass.
                                   Defaults to 3.0.
            port (int, optional): Control port, useful only for rank-0. Defaults to 65535.
            prefill (int, optional): Howmany Events to pre-populate. Defaults to 1024.
            enabled (bool, optional): Whether or not collection is enabled on startup.
                                      Defaults to False.
        """
        if StragglerDetector._configured:
            # don't throw
            return
        log.info("successfully entered intstantiation of Straggler Detectior")
        StragglerDetector._configured = True
        self.bdata = False
        self.start = self.null_method
        self.stop = self.null_method
        self._off = True
        # No CUDA, No Support
        if torch.cuda.is_available():
            self._off = not enabled
            self.world = world
            self.rank = rank
            self.mmcnt = mmcnt if mmcnt > 1 else 1
            self.amp = amp
            self.port = port
            self.toggle = False
            self.bdata = False
            self.idx = 0
            self.idx_q = queue.LifoQueue()
            self.evt_q = queue.LifoQueue()
            self.start_events = []
            self.stop_events = []
            self.start_time = []
            self.stop_time = []
            self.start_batch = []
            self.stop_batch = []
            backend = torch.distributed.get_backend()
            if backend == "nccl":
                self.dev = torch.cuda.current_device()
            else:
                self.dev = torch.device("cpu")
            # cache some events
            for _ in range(prefill):
                self.evt_q.put(torch.cuda.Event(enable_timing=True))
            if self.rank == 0:
                # Start the controller
                self._controller()
            if not self._off:
                log.info("successfully defined self.start")
                self.start = self.start_method
                self.stop = self.stop_method

    def reset(self) -> None:
        """This method is called to reset the metrics state of the instance
        It is generally called from within elapsed() after extracting per rank metrics.
        """
        if self._off:
            return
        self.idx = 0
        self.idx_q = queue.LifoQueue()
        # Pool them
        _ = [self.evt_q.put(ev) for ev in self.start_events]
        _ = [self.evt_q.put(ev) for ev in self.stop_events]
        self.start_events = []
        self.stop_events = []
        # Use regular timers
        self.start_time = []
        self.stop_time = []
        self.start_batch = []
        self.stop_batch = []
        self.bdata = False

    def start_method(self) -> None:
        """This method adds the start timers.
        Both cuda event and perf_counter are added. If bdata is set to
        true from __call__, this method skips inserting cuda
        timer. This way it can be used to measure time spent on
        CPU - generally useful for timing get_batch()
        """
        # Not reentrant
        # First check if this start is for data
        log.info("start method called")
        if self.bdata:
            self.start_batch.append(time.perf_counter_ns())
            self.stop_batch.append(0)  # this indicate we need to add timer
            self.bdata = False
            return
        if self.evt_q.qsize() > 1:
            sev = self.evt_q.get()  # no try-catch
            eev = self.evt_q.get()  # no try-catch
        else:
            sev = torch.cuda.Event(enable_timing=True)
            eev = torch.cuda.Event(enable_timing=True)
        self.start_events.append(sev)
        self.stop_events.append(eev)
        self.start_time.append(0)
        self.stop_time.append(0)
        self.idx_q.put(self.idx)
        self.start_time[self.idx] = time.perf_counter_ns()
        self.start_events[self.idx].record()
        self.idx += 1
        log.info("start method finished")

    def stop_method(self) -> None:
        """This method adds the stop timers.
        Both cuda event and perf_counter are added. If bdata is set to
        true from __call__, this method skips inserting cuda
        timer. Also see start_method()
        """
        # Not reentrant
        # First check if this stop is for data
        dle = len(self.stop_batch) - 1
        if dle >= 0 and self.stop_batch[dle] == 0:
            self.stop_batch[dle] = time.perf_counter_ns()
            return
        idx = self.idx_q.get()
        self.stop_time[idx] = time.perf_counter_ns()
        self.stop_events[idx].record()

    def elapsed(self) -> Tuple[float, float, int, int, int, int]:
        """This method is called from report(), or can be called directly
         It is called to collect all the elapsed time since last reset().
         It finally calls reset()
        Returns:
            Tuple[float, float, int, int, int, int]: see below for returns
                delta       : time spent in kernel
                batch_delta : time spent in get_batch
                temp        : observed gpu temp
                power       : observed gpu power
                util        : observed gpu utilization
                clock       : observed gpu clock
        """
        if self._off:
            # match with return below
            #log.info("elapsed is off")
            return 0, 0, 0, 0, 0, 0
        ls_ev = len(self.start_events)
        le_ev = len(self.stop_events)
        ls_bs = len(self.start_batch)
        ls_be = len(self.stop_batch)

        log.info("length of start events: " + str(ls_ev))
        delta = 0.0
        batch_delta = 0.0
        temp = 0
        power = 0
        clock = 0
        if ls_ev != le_ev:
            self.logger.warning(f"Event Start/Stop out of sync {ls_ev}/{le_ev}")
        elif ls_bs != ls_be:
            self.logger.warning(f"get_batch Start/Stop out of sync {ls_bs}/{ls_be}")
        else:
            temp = torch.cuda.temperature()
            power = torch.cuda.power_draw()
            util = torch.cuda.utilization()
            clock = torch.cuda.clock_rate()
            torch.cuda.synchronize()
            # Process Events
            for i in range(ls_ev):
                e_ev = self.start_events[i].elapsed_time(self.stop_events[i])
                e_tm = (self.stop_time[i] - self.start_time[i]) / 1e6  # ns to ms
                # Pick the larger of Event and perf_counter time?
                delta += max(e_ev, e_tm)
            # Process get_batch
            for i in range(ls_bs):
                log.info("getting batch")
                batch_delta = (self.stop_batch[i] - self.start_batch[i]) / 1e3  # us
        self.reset()  # Prepare for next round
        # time in ms, batch_delta in us, check return above
        return delta, batch_delta, temp, power, util, clock

    def report(self, total_flops: float = 0.0, log_interval: int = 0) -> bool:
        """Function to log the min/max metircs and the associated rank over a time period
        It finds the slowest and fastest rank among all ranks. It should be
        called by all ranks, but only rank-0 prints the analysis
        At the end it checks, if the straggler detector should
        remain active or if it should be deactivated.
        Args:
            total_flops (float, optional): The theoretical flops over the period. Defaults to 0.0.
            log_interval (int, optional): The training interval over which reporting is called(ms)
                                          Defaults to 0.
        Returns:
            bool: True if reported, else False
        """
        ret = False
        if not self._off and total_flops > 0.0 and log_interval > 0:
            elapsed, btime_us, temp, power, util, clock = self.elapsed()  # get raw time
            log.info("elapsed: " + str(elapsed))
            ptime = elapsed / (log_interval * 1.0)  # avg per iteration elapsed time, ms
            btime = btime_us / (log_interval * 1.0)  # avg per iteration get_batch time, us
            if btime < 1e-8:
                log.info("BATCH TIME IS GENUINELY 0")
            api_flops = total_flops / (log_interval * 1.0)  # avg per iteration flops, ms
            """
            apir_flops = api_flops / (
                ptime * 10 ** 9 * self.world
            )  # this is avg per iteration this rank's thruput, TFLOP/s (note 10**9),
            """
            apir_flops = api_flops / (
                ptime * 10 ** 9
            )
            et_flops = apir_flops / self.amp  # Estimated TFLOPs, not tracing backward

            o_dt = self._min_max(
                ptime, btime, float(temp), float(power), float(util), float(clock), et_flops,
            )
            if self.rank == 0:
                now = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                min_flops, min_frank, _ = o_dt.aflops[0]()
                max_flops, max_frank, _ = o_dt.aflops[-1]()
                self.logger.info(
                    f"{now} | "
                    f"MnRtt/Rnk: {o_dt.min_elapsed} | "
                    f"MxRtt/Rnk: {o_dt.max_elapsed} | "
                    f"MnPwr/Rnk: {o_dt.min_power} | "
                    f"MxPwr/Rnk: {o_dt.max_power} | "
                    f"MnTmp/Rnk: {o_dt.min_temp} | "
                    f"MxTmp/Rnk: {o_dt.max_temp} | "
                    f"MnUtl/Rnk: {o_dt.min_util} | "
                    f"MxUtl/Rnk: {o_dt.max_util} | "
                    f"MnClk/Rnk: {o_dt.min_clock} | "
                    f"MxClk/Rnk: {o_dt.max_clock} | "
                    f"MnDRtt/Rnk: {o_dt.min_btime} | "
                    f"MxDRtt/Rnk: {o_dt.max_btime} | "
                    f"MnEtpt/Rnk: {min_flops:.2f}TF/{min_frank} | "
                    f"MxEtpt/Rnk: {max_flops:.2f}TF/{max_frank}"
                )
                if self.mmcnt > 1 and self.mmcnt < self.world:
                    line = f"^^^^ Bottom {self.mmcnt} Ranks with lowest  Etpt(TF):"
                    for i in range(self.mmcnt):
                        line += f" {o_dt.aflops[i]},"
                    self.logger.info(line)
                    line = f"^^^^ Top    {self.mmcnt} Ranks with highest Etpt(TF):"
                    shift = self.world - self.mmcnt
                    for i in range(self.mmcnt):
                        line += f" {o_dt.aflops[i+shift]},"
                    self.logger.info(line)
                ret = True

        # Check/Communicate if tracking is turned off or on
        self._check_toggle()
        return ret

    def _check_toggle(self) -> None:
        """Helper method to check if a request to toggle the collection state was made
        It checks iof collection state toggle req was made via the server listening on
        rank-0 since last call to report(). Called by report(). Calling this method
        indirectly from report() is the only way to activate the change that is made
        via rank-0
        """
        # If no change just commnunicate the current
        off = self._off
        if self.rank == 0 and self.toggle:
            off = not self._off
            self.toggle = False
        state = torch.tensor(off, dtype=torch.bool, device=self.dev)
        torch.distributed.broadcast(state, 0)  # Blocking
        self._off = state.item()
        if not self._off:
            self.start = self.start_method
            self.stop = self.stop_method
            state = "ON"
        else:
            self.start = self.null_method
            self.stop = self.null_method
            state = "OFF"
        if self.rank == 0 and off is not self._off:
            self.logger.info(f"Toggling StragglerDetector State {state}")

    def _handler(self) -> None:
        """Thread function for the controller.
        It is a tcp-server that listens on a port. Uses HTTP protocol.
        If connected to it using curl, it indicates a toggle of the
        collection state. The actual toggling happens at the end of
        calling report() when _check_toggle() is called.
        """
        resp = f"HTTP/1.0 200 OK\r\nConnection: Close\r\nContent-length: "

        if self.rank == 0:
            state = "OFF" if self._off else "ON"
            self.logger.info(
                f"Controller ready to recv " f"commands on port {self.port}. Current state {state}"
            )
            while True:
                try:
                    conn, _ = self.sock.accept()
                    _ = conn.recv(1024)
                    self.toggle = True
                    state = "ON" if self._off else "OFF"
                    msg = f"Will turn StragglerDetector {state} at next logging interval"
                    msg_len = len(msg)
                    final_resp = f"{resp}{msg_len}\r\n\r\n{msg}"
                    conn.send(final_resp.encode())
                    conn.close()
                    self.logger.info(msg)
                except Exception as err:
                    self.logger.error(f"Error in stragler handler.. {str(err)}")
                    return

    def _controller(self):
        """Installs a controller listener that is used to toggle collection state.
        Called from configure(). Ignored for all ranks other than rank-0
        """
        try:
            if self.rank == 0:
                neth = "0.0.0.0"
                netp = self.port
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.bind((neth, netp))
                self.sock.listen(128)
                self.ctrlr = threading.Thread(
                    target=self._handler, args=(), name="straggler", daemon=True
                )
                self.ctrlr.start()
        except Exception as err:
            self.logger.warning(f"StragglerDetector cannot be controlled.. {str(err)}")

    def _min_max(
        self,
        ptime: float,
        btime: float,
        temp: float,
        power: float,
        util: float,
        clock: float,
        flops: float,
    ) -> Union[_StragglerData, None]:
        """Helper function to find the min/max values
        Args:
            ptime (float): avg per iteration gpu time
            btime (float): avg per iteration cpu time
            temp (float): gpu temp at the time of reporting
            power (float): gpu power at the time of reporting
            util (float): gpu util at the time of reporting
            clock (float): gpu clock at the time of reporting
            flops (float): estimated flops for the rank
        Returns:
            Union[_StragglerData, None]: It contains the min/max of few metrics and the
                                         corresponding rank it also has sorted list of
                                         all (flops, rank) sorted by flops (aflops)
                                         or returns None if collecton is disabled
        """
        if self._off:
            return None
        # initialize output data object
        o_dt = _StragglerData()

        prof_data = {}
        prof_data["rank"] = self.rank
        prof_data["time"] = ptime
        prof_data["btime"] = btime
        prof_data["temp"] = temp
        prof_data["power"] = power
        prof_data["util"] = util
        prof_data["clock"] = clock
        prof_data["flops"] = flops

        if self.rank == 0:
            data_list = [prof_data] * self.world
        else:
            data_list = None

        # this is blocking by default
        torch.distributed.gather_object(prof_data, object_gather_list=data_list, dst=0)

        if self.rank == 0:
            min_ctime = min(data_list, key=lambda k: k["time"])  # elapsed
            max_ctime = max(data_list, key=lambda k: k["time"])  # elapsed

            min_cbatch = min(data_list, key=lambda k: k["btime"])  # batch time
            max_cbatch = max(data_list, key=lambda k: k["btime"])  # batch time

            min_ctemp = min(data_list, key=lambda k: k["temp"])  # temp
            max_ctemp = max(data_list, key=lambda k: k["temp"])  # temp

            min_cpower = min(data_list, key=lambda k: k["power"])  # power
            max_cpower = max(data_list, key=lambda k: k["power"])  # power

            min_cutil = min(data_list, key=lambda k: k["util"])  # gpu util
            max_cutil = max(data_list, key=lambda k: k["util"])  # gpu util

            min_cclock = min(data_list, key=lambda k: k["clock"])  # gpu clock
            max_cclock = max(data_list, key=lambda k: k["clock"])  # gpu clock

            min_val = min_ctime["time"]
            min_rank = min_ctime["rank"]
            max_val = max_ctime["time"]
            max_rank = max_ctime["rank"]
            o_dt.min_elapsed = _ValueWithRank(min_val, min_rank, "ms")
            o_dt.max_elapsed = _ValueWithRank(max_val, max_rank, "ms")

            min_val = min_cbatch["btime"]
            min_rank = min_cbatch["rank"]
            max_val = max_cbatch["btime"]
            max_rank = max_cbatch["rank"]
            o_dt.min_btime = _ValueWithRank(min_val, min_rank, "us")
            o_dt.max_btime = _ValueWithRank(max_val, max_rank, "us")

            min_val = min_ctemp["temp"]
            min_rank = min_ctemp["rank"]
            max_val = max_ctemp["temp"]
            max_rank = max_ctemp["rank"]
            o_dt.min_temp = _ValueWithRank(min_val, min_rank, "C")
            o_dt.max_temp = _ValueWithRank(max_val, max_rank, "C")

            min_val = min_cpower["power"]
            min_rank = min_cpower["rank"]
            max_val = max_cpower["power"]
            max_rank = max_cpower["rank"]
            o_dt.min_power = _ValueWithRank(min_val, min_rank, "W")
            o_dt.max_power = _ValueWithRank(max_val, max_rank, "W")

            min_val = min_cutil["util"]
            min_rank = min_cutil["rank"]
            max_val = max_cutil["util"]
            max_rank = max_cutil["rank"]
            o_dt.min_util = _ValueWithRank(min_val, min_rank, "%")
            o_dt.max_util = _ValueWithRank(max_val, max_rank, "%")

            min_val = min_cclock["clock"]
            min_rank = min_cclock["rank"]
            max_val = max_cclock["clock"]
            max_rank = max_cclock["rank"]
            o_dt.min_clock = _ValueWithRank(min_val, min_rank, "MHz")
            o_dt.max_clock = _ValueWithRank(max_val, max_rank, "MHz")

            o_dt.aflops = [
                _ValueWithRank(d.get("flops"), d.get("rank")) for _, d in enumerate(data_list)
            ]
            o_dt.aflops.sort(key=lambda val_with_rank: val_with_rank()[0])
        # wait for everyone here
        torch.distributed.barrier()

        return o_dt

    @property
    def enabled(self) -> bool:
        """Can be called to check the enabled state of the instance
        Note:
            After the request to toggle the state, the
            actual state change happens at end of call
            to report()
        """
        return not self._off

    @property
    def configured(self) -> bool:
        """Can be called to check if the the instance is already configured
        Returns:
            bool: returns True if configure was called and was a success, else False
        """
        return StragglerDetector._configured

    @property
    def my_rank(self):
        """Can be called to get configured rank of this instance
        Returns:
            int: Configured rank for this instance
        """
        return self.rank

    @property
    def world_size(self) -> int:
        """Can be called to get configured world of this instance
        Returns:
            int: World size configured for this instance
        """
        return self.world

    def null_method(self) -> None:
        """Default method to initialize start/stop method ptrs"""
        pass

    def __enter__(self) -> "StragglerDetector":
        """Define context/instance entry
        Returns:
            StragglerDetector: the instance
        """
        self.start()
        return self

    def __call__(self, bdata: bool = False) -> "StragglerDetector":
        """Callable for the instance. Set context state,
        Useful when the context is used for cpu timers only when bdata=True
        Args:
            bdata (bool, optional): when true, only enables cpu timers. Defaults to False.
        Returns:
            StragglerDetector: the instance
        """
        self.bdata = bdata
        return self

    def __exit__(
        self,
        ex_type: Optional[Type[BaseException]],
        ex_val: Optional[BaseException],
        ex_tb: Optional[TracebackType],
    ) -> bool:
        """Define context/instance exit, calls the stop method
        Args:
            ex_type (Optional[Type[BaseException]]): Exception type
            ex_val (Optional[BaseException]): _description_
            ex_tb (Optional[TracebackType]): _description_
        Returns:
            bool: True if the exception was handled
        """
        # Should not suppress errors even if turned off
        ret = False
        if ex_type is not None:
            err = traceback.format_exception(ex_tb)
            self.logger.warning(f"{str(ex_val)}\n{err}")
            ret = True
        self.stop()
        return ret



class GlobalStragglerDetector(Callback):

    def __init__(self) -> None:
        self.stimer = None
        self.log_interval = 0
        #self.start_time = None

    def init(self, state: State, logger: Logger) -> None:
        self.stimer = StragglerDetector()
        port = int(os.environ.get('MASTER_PORT'))
        rank = dist.get_global_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            self.stimer.configure(world_size, rank, enabled=True, port=port, amp=1.0)
        else:
            self.stimer.configure(world_size, rank, enabled=True, amp=1.0)
        

    def batch_start(self, state: State, logger: Logger):
        #self.start_time = time.time()
        self.stimer.start()

    def batch_end(self, state: State, logger: Logger):
        # Calculate duration of the current batch
        #batch_time = (time.time() - self.start_time) * 1000
        #self.log_interval = int(batch_time)

        #log.info("log_interval:" + str(self.log_interval))
        # Compute flops stats if model has flops_per_batch
        composer_model = state.model
        if not isinstance(composer_model, ComposerModel):
            composer_model = composer_model.module
        if hasattr(composer_model, 'flops_per_batch'):
            model_flops_per_batch = composer_model.flops_per_batch  # type: ignore
            if not isinstance(model_flops_per_batch, Callable):
                raise TypeError(
                    'flops_per_batch must a callable accepting a batch and '
                    f'returning an int or float. Instead, got {type(model_flops_per_batch)}.',
                )
            device_flops_per_batch = model_flops_per_batch(state.batch)
            log.info("StragglerDetector Flops: " + str(device_flops_per_batch))
            self.stimer.stop()
            #self.stimer.report(total_flops=device_flops_per_batch, log_interval=self.log_interval)
            self.stimer.report(total_flops=device_flops_per_batch, log_interval=1)
            
           

        else:
            raise ValueError("The 'flops_per_batch' attribute is not present in this model; StragglerDetector requires tracking flops per batch.")

        









