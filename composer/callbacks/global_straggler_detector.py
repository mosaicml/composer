# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0s

"""
Monitor straggler metrics during training.

Original StragglerDetector README: https://github.com/NVIDIA/Megatron-LM/blob/c4d12e26b2dc25a2eab7da92e2ac30338c0ed3de/megatron/core/README_STRAGGLER.md
Original StragglerDetector implementation: https://github.com/NVIDIA/Megatron-LM/blob/c4d12e26b2dc25a2eab7da92e2ac30338c0ed3de/megatron/core/utils.py
"""

import logging
import queue
import sys
import time
import traceback
from dataclasses import dataclass, field
from types import TracebackType
from typing import Callable, List, Optional, Tuple, Type, Union

import torch

from composer.core import Callback, State
from composer.loggers import Logger
from composer.models.base import ComposerModel
from composer.utils import dist

__all__ = ['GlobalStragglerDetector']


class _ValueWithRank:
    """_ValueWithRank is an internal class, not for use outside this module.

    Attributes:
        _rank (int): rank for the value
        _value (float) : the value it stores, eg elapsed time
        _unit (str) : unit for the value
    """

    def __init__(self, value: float, rank: int, unit: str = '') -> None:
        """Initializer.

        Args:
            value (float): the initial value with which it is inited
            rank (int): the rank number
            unit (str) : the unit of the value, eg ms or flops
        """
        self._rank = rank
        self._value = value
        self._unit = unit

    def __lt__(self, other) -> bool:
        """Check if value of self is smaller than other's value.

        Args:
            other (_ValueWithRank): The other object to compare with
        Returns:
            bool: True if lhs._value of operand is less than rhs._value, else False
        """
        return self._value < other._value

    def __gt__(self, other) -> bool:
        """Check if value of self is larger than other's value.

        Args:
            other (_ValueWithRank): The other object to compare with

        Returns:
            bool: True if lhs._value of operand is greater than rhs._value, else False
        """
        return self._value > other._value

    def __call__(self) -> Tuple[float, int, str]:
        """Returns the value, the rank, and unit as a Tuple.

        Returns:
            Tuple[float, int, str]: value, rank, unit
        """
        return self._value, self._rank, self._unit

    #edited __str__ to include the word "Rank" for clarity
    def __str__(self) -> str:
        """String representation of the object.

        Returns:
            str: strigified object
        """
        return f"{self._value:.2f}{self._unit}/Rank-{self._rank}"


@dataclass
class _StragglerData:
    """_StragglerData is an internal dataclass, not for use outside this module.

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
    min_elapsed = _ValueWithRank(sys.float_info.max, 0, 'ms')
    max_elapsed = _ValueWithRank(sys.float_info.min, 0, 'ms')
    # get_batch time
    min_btime = _ValueWithRank(sys.float_info.max, 0, 'us')
    max_btime = _ValueWithRank(sys.float_info.min, 0, 'us')
    # temp
    min_temp = _ValueWithRank(sys.float_info.max, 0, 'C')
    max_temp = _ValueWithRank(sys.float_info.min, 0, 'C')
    # power
    min_power = _ValueWithRank(sys.float_info.max, 0, 'W')
    max_power = _ValueWithRank(sys.float_info.min, 0, 'W')
    # util
    min_util = _ValueWithRank(sys.float_info.max, 0, '%')
    max_util = _ValueWithRank(sys.float_info.min, 0, '%')
    # clock
    min_clock = _ValueWithRank(sys.float_info.max, 0, 'MHz')
    max_clock = _ValueWithRank(sys.float_info.min, 0, 'MHz')
    aflops: List[_ValueWithRank] = field(default_factory=list)


class StragglerDetector:
    """Singleton Class implementing per rank Straggler Detector.

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
    _instance = None
    

    def __new__(cls: Type['StragglerDetector'], *args, **kwargs) -> 'StragglerDetector':
        """Constructor.

        Creates an instance of the class if not created

        Args:
            cls (Type[&#39;StragglerDetector&#39;]): The class type
        Returns:
            StragglerDetector: the class instance
        """
        if cls._instance is None:
            cls._instance = super(StragglerDetector, cls).__new__(cls)
            cls._instance.__initialized = False 
        return cls._instance

    def __init__(self, world: int, rank: int, mmcnt: int = 1, amp: float = 1.0, prefill: int = 1024) -> None:
        """Initializer.

        The inital state of the StragglerDetector instance is disabled.
        The enabled state is indicated using self._off member variable
        and the property enabled.
        """
        if self.__initialized:
            return

        self.__initialized = True
        self.logger = logging.getLogger(__name__)
        self.world = world
        self.rank = rank
        self.mmcnt = mmcnt
        self.amp = amp
        self.bdata = False
        self.idx = 0

        self.start_events = []
        self.stop_events = []
        self.start_time = []
        self.stop_time = []
        self.start_batch = []
        self.stop_batch = []
        self.idx_q = queue.LifoQueue()
        self.evt_q = queue.LifoQueue()

        if torch.cuda.is_available():
            self.dev = torch.cuda.current_device()
        else:
            self.dev = torch.device('cpu')
        for _ in range(prefill):
            self.evt_q.put(torch.cuda.Event(enable_timing=True))

        self.start = self.start_method
        self.stop = self.stop_method
        self.__initialized = True

    def reset(self) -> None:
        """Reset is called to reset the metrics state of the instance.

        It is generally called from within elapsed() after extracting per rank metrics.
        """
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
        """start_method adds the start timers.

        Both cuda event and perf_counter are added. If bdata is set to
        true from __call__, this method skips inserting cuda
        timer. This way it can be used to measure time spent on
        CPU - generally useful for timing get_batch()
        """
        # Not reentrant
        # First check if this start is for data
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

    def stop_method(self) -> None:
        """stop_method adds the stop timers.

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
        """Elapsed is called from report(), or can be called directly.

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
        ls_ev = len(self.start_events)
        le_ev = len(self.stop_events)
        ls_bs = len(self.start_batch)
        ls_be = len(self.stop_batch)

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
                batch_delta = (self.stop_batch[i] - self.start_batch[i]) / 1e3  # us
        self.reset()  # Prepare for next round
        # time in ms, batch_delta in us, check return above
        return delta, batch_delta, temp, power, util, clock

    # Modified following method from original Megatron-LM
    def report(self, total_flops: float = 0.0, log_interval: int = 0) -> Tuple[bool, dict]:
        """Function to log the min/max metircs and the associated rank over a time period.

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
            dict: Dict of min/max metrics and their associated ranks, empty if not rank-0
        """
        self.logger.info("Entered report")
        ret = False
        min_max_data = {}
        if total_flops > 0.0 and log_interval > 0:
            elapsed, btime_us, temp, power, util, clock = self.elapsed()  # get raw time
            ptime = elapsed / (log_interval * 1.0)  # avg per iteration elapsed time, ms
            btime = btime_us / (log_interval * 1.0)  # avg per iteration get_batch time, us
            api_flops = total_flops / (log_interval * 1.0)  # avg per iteration flops, ms

            apir_flops = api_flops / (ptime * 10**9)
            et_flops = apir_flops / self.amp  # Estimated TFLOPs, not tracing backward

            o_dt = self._min_max(
                ptime,
                btime,
                float(temp),
                float(power),
                float(util),
                float(clock),
                et_flops,
            )
            if self.rank == 0:
                min_flops, min_frank, _ = o_dt.aflops[0]()
                max_flops, max_frank, _ = o_dt.aflops[-1]()

                min_throughput = _ValueWithRank(min_flops, min_frank, 'TF')
                max_throughput = _ValueWithRank(max_flops, max_frank, 'TF')

                min_max_data = {
                    'MinRoundTripTime/Rank': o_dt.min_elapsed,
                    'MaxRoundTripTime/Rank': o_dt.max_elapsed,
                    'MinPower/Rank': o_dt.min_power,
                    'MaxPower/Rank': o_dt.max_power,
                    'MinTemp/Rank': o_dt.min_temp,
                    'MaxTemp/Rank': o_dt.max_temp,
                    'MinUtilization/Rank': o_dt.min_util,
                    'MaxUtilization/Rank': o_dt.max_util,
                    'MinClock/Rank': o_dt.min_clock,
                    'MaxClock/Rank': o_dt.max_clock,
                    'MinBatchLoadLatency/Rank': o_dt.min_btime,
                    'MaxBatchLoadLatency/Rank': o_dt.max_btime,
                    'MinThroughput/Rank': min_throughput,
                    'MaxThroughput/Rank': max_throughput,
                }

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

        return ret, min_max_data

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
        """Helper function to find the min/max values.

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
        # initialize output data object
        o_dt = _StragglerData()

        prof_data = {}
        prof_data['rank'] = self.rank
        prof_data['time'] = ptime
        prof_data['btime'] = btime
        prof_data['temp'] = temp
        prof_data['power'] = power
        prof_data['util'] = util
        prof_data['clock'] = clock
        prof_data['flops'] = flops

        if self.rank == 0:
            data_list = [prof_data] * self.world
        else:
            data_list = None

        # this is blocking by default
        torch.distributed.gather_object(prof_data, object_gather_list=data_list, dst=0)

        if self.rank == 0:
            min_ctime = min(data_list, key=lambda k: k['time'])  # elapsed
            max_ctime = max(data_list, key=lambda k: k['time'])  # elapsed

            min_cbatch = min(data_list, key=lambda k: k['btime'])  # batch time
            max_cbatch = max(data_list, key=lambda k: k['btime'])  # batch time

            min_ctemp = min(data_list, key=lambda k: k['temp'])  # temp
            max_ctemp = max(data_list, key=lambda k: k['temp'])  # temp

            min_cpower = min(data_list, key=lambda k: k['power'])  # power
            max_cpower = max(data_list, key=lambda k: k['power'])  # power

            min_cutil = min(data_list, key=lambda k: k['util'])  # gpu util
            max_cutil = max(data_list, key=lambda k: k['util'])  # gpu util

            min_cclock = min(data_list, key=lambda k: k['clock'])  # gpu clock
            max_cclock = max(data_list, key=lambda k: k['clock'])  # gpu clock

            min_val = min_ctime['time']
            min_rank = min_ctime['rank']
            max_val = max_ctime['time']
            max_rank = max_ctime['rank']
            o_dt.min_elapsed = _ValueWithRank(min_val, min_rank, 'ms')
            o_dt.max_elapsed = _ValueWithRank(max_val, max_rank, 'ms')

            min_val = min_cbatch['btime']
            min_rank = min_cbatch['rank']
            max_val = max_cbatch['btime']
            max_rank = max_cbatch['rank']
            o_dt.min_btime = _ValueWithRank(min_val, min_rank, 'us')
            o_dt.max_btime = _ValueWithRank(max_val, max_rank, 'us')

            min_val = min_ctemp['temp']
            min_rank = min_ctemp['rank']
            max_val = max_ctemp['temp']
            max_rank = max_ctemp['rank']
            o_dt.min_temp = _ValueWithRank(min_val, min_rank, 'C')
            o_dt.max_temp = _ValueWithRank(max_val, max_rank, 'C')

            min_val = min_cpower['power']
            min_rank = min_cpower['rank']
            max_val = max_cpower['power']
            max_rank = max_cpower['rank']
            o_dt.min_power = _ValueWithRank(min_val, min_rank, 'W')
            o_dt.max_power = _ValueWithRank(max_val, max_rank, 'W')

            min_val = min_cutil['util']
            min_rank = min_cutil['rank']
            max_val = max_cutil['util']
            max_rank = max_cutil['rank']
            o_dt.min_util = _ValueWithRank(min_val, min_rank, '%')
            o_dt.max_util = _ValueWithRank(max_val, max_rank, '%')

            min_val = min_cclock['clock']
            min_rank = min_cclock['rank']
            max_val = max_cclock['clock']
            max_rank = max_cclock['rank']
            o_dt.min_clock = _ValueWithRank(min_val, min_rank, 'MHz')
            o_dt.max_clock = _ValueWithRank(max_val, max_rank, 'MHz')

            o_dt.aflops = [_ValueWithRank(d.get('flops'), d.get('rank')) for _, d in enumerate(data_list)]
            o_dt.aflops.sort(key=lambda val_with_rank: val_with_rank()[0])
        # wait for everyone here
        torch.distributed.barrier()

        return o_dt

    def __enter__(self) -> 'StragglerDetector':
        """Define context/instance entry.

        Returns:
            StragglerDetector: the instance
        """
        self.start()
        return self

    def __call__(self, bdata: bool = False) -> 'StragglerDetector':
        """Callable for the instance. Set context state.

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
        """Define context/instance exit, calls the stop method.

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
    """Logs the minimum and maximum training values across all ranks for the following metrics.

        RoundTripTime: Time spent in all the traced ops in the current batch
        Power: GPU Power Consumption
        Temp: GPU Temperature
        Utilization: GPU Utilization
        Clock: GPU Clock
        BatchLoadLatency: Time spent loading the current batch from the dataset
        Throughput: Estimated throughput for the current batch

    The maximum and minimum values for these metrics, alongside their respective ranks, are logged
    on the :attr:`.Event.BATCH_END` event for every batch.

    To compute `flops_per_sec`, the model attribute `flops_per_batch` should be set to a callable
    which accepts a batch and returns the number of flops for that batch. Typically, this should
    be flops per sample times the batch size unless pad tokens are used.

    The wall clock time is logged on every :attr:`.Event.BATCH_END` event.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import GlobalStragglerDetector
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration='1ep',
            ...     callbacks=[GlobalStragglerDetector()],
            ... )

    The metrics are logged by the :class:`.Logger` to the following keys as
    described below.

    +-------------------------------------+-----------------------------------------------------------+
    | Key                                 | Logged data                                               |
    +=====================================+===========================================================+
    |                                     | Minimum time spent in all the traced ops in the           |
    | `MinRoundTripTime/Rank`             | current batch across all ranks for the corresponding rank |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Maximum time spent in all the traced ops in the           |
    | `MaxRoundTripTime/Rank`             | current batch across all ranks for the corresponding rank |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    | `MinPower/Rank`                     | Minimum GPU Power consumed for the corresponding rank     |
    +-------------------------------------+-----------------------------------------------------------+
    | `MaxPower/Rank`                     | Maximum GPU Power consumed for the corresponding rank     |
    +-------------------------------------+-----------------------------------------------------------+
    | `MinTemp/Rank`                      | Minimum GPU Temperature for the corresponding rank        |
    +-------------------------------------+-----------------------------------------------------------+
    | `MaxTemp/Rank`                      | Maximum GPU Temperature for the corresponding rank        |
    +-------------------------------------+-----------------------------------------------------------+
    | `MinUtilization/Rank`               | Minimum GPU Utilization for the corresponding rank        |
    +-------------------------------------+-----------------------------------------------------------+
    | `MaxUtilization/Rank`               | Maximum GPU Utilization for the corresponding rank        |
    +-------------------------------------+-----------------------------------------------------------+
    | `MinClock/Rank`                     | Minimum GPU Clock for the corresponding rank              |
    +-------------------------------------+-----------------------------------------------------------+
    | `MaxClock/Rank`                     | Maximum GPU Clock for the corresponding rank              |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Minimum time spent loading the current batch from the     |
    | `MinBatchLoadLatency/Rank`          | dataset across all ranks for the corresponding rank       |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Maximum time spent loading the current batch from the     |
    | `MaxBatchLoadLatency/Rank`          | dataset across all ranks for the corresponding rank       |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    | `MinThroughput/Rank`                | Minimum estimated throughput for the corresponding rank   |
    +-------------------------------------+-----------------------------------------------------------+
    | `MaxThroughput/Rank`                | Maximum estimated throughput for the corresponding rank   |
    +-------------------------------------+-----------------------------------------------------------+


    Args:
        None
    """

    def __init__(self) -> None:
        self.off = False

    def init(self, state: State, logger: Logger) -> None:
        print("entered init")
        rank = dist.get_global_rank()
        world_size = dist.get_world_size()
        self.stimer = StragglerDetector(world_size, rank)
        print("finished init")

    def batch_start(self, state: State, logger: Logger):
        if self.off:
            return
        self.stimer.start()

    def batch_end(self, state: State, logger: Logger):
        print("entered batch_end")
        if self.off:
            return
        # Compute flops stats if model has flops_per_batch
        composer_model = state.model
        if not isinstance(composer_model, ComposerModel):
            composer_model = composer_model.module
        if hasattr(composer_model, 'flops_per_batch'):
            model_flops_per_batch = composer_model.flops_per_batch  # type: ignore
            if not isinstance(model_flops_per_batch, Callable):
                self.off = True
                print(
                    'Model must contain the parameter model_flops_per_batch for throughput calculation and be Callable. Turning off GlobalStragglerDetector Callback.',
                )
                return
            device_flops_per_batch = model_flops_per_batch(state.batch)
            self.stimer.stop()
            is_rank_zero, min_max_data = self.stimer.report(total_flops=device_flops_per_batch, log_interval=1)
            if is_rank_zero:
                logger.log_metrics(min_max_data)

        else:
            self.off = True
            print(
                'Model must contain the parameter model_flops_per_batch for throughput calculation. Turning off GlobalStragglerDetector Callback.',
            )
            return

    def before_dataloader(self, state: State, logger: Logger):
        if self.off:
            return
        self.stimer.bdata = True
        self.stimer.start()

    def after_dataloader(self, state: State, logger: Logger):
        if self.off:
            return
        self.stimer.stop()
        self.stimer.report()
        self.stimer.bdata = False
