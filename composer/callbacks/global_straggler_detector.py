"""
from megatron.core.utils import StragglerDetector
from composer.core import Callback, State, Event, Time
from composer.utils import dist
from typing import List, Union
from dataclasses import dataclass


__all__ = ["globalStragglerDetector"]

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
class _AllStragglerData:
    """Custom class to store data for all ranks.

    Attributes:
        elapsed_times (List[_ValueWithRank]): List of elapsed times for all ranks
        batch_times (List[_ValueWithRank]): List of batch times for all ranks
        temperatures (List[_ValueWithRank]): List of GPU temperatures for all ranks
        power_draws (List[_ValueWithRank]): List of GPU power draws for all ranks
        utilizations (List[_ValueWithRank]): List of GPU utilizations for all ranks
        clock_rates (List[_ValueWithRank]): List of GPU clock rates for all ranks
        estimated_flops (List[_ValueWithRank]): List of estimated flops for all ranks
    """

    elapsed_times: List[_ValueWithRank] = None
    batch_times: List[_ValueWithRank] = None
    temperatures: List[_ValueWithRank] = None
    power_draws: List[_ValueWithRank] = None
    utilizations: List[_ValueWithRank] = None
    clock_rates: List[_ValueWithRank] = None
    estimated_flops: List[_ValueWithRank] = None


class reportAllStragglerDetector(StragglerDetector):
    


    def _get_all_ranks(
        self,
        ptime: float,
        btime: float,
        temp: float,
        power: float,
        util: float,
        clock: float,
        flops: float,
    ) -> Union[_AllStragglerData, None]:
        """Helper function to collect all values for all ranks

        Args:
            ptime (float): avg per iteration gpu time
            btime (float): avg per iteration cpu time
            temp (float): gpu temp at the time of reporting
            power (float): gpu power at the time of reporting
            util (float): gpu util at the time of reporting
            clock (float): gpu clock at the time of reporting
            flops (float): estimated flops for the rank

        Returns:
            Union[_StragglerData, None]: It contains all values for all ranks
        """
        if self._off:
            return None

        all_data = _AllStragglerData()
        

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

        # Gather data using distributed gather
        torch.distributed.gather_object(prof_data, object_gather_list=data_list, dst=0)

        if self.rank == 0:
            all_data.elapsed_times = [_ValueWithRank(d["time"], d["rank"], "ms") for d in data_list]
            all_data.batch_times = [_ValueWithRank(d["btime"], d["rank"], "us") for d in data_list]
            all_data.temperatures = [_ValueWithRank(d["temp"], d["rank"], "C") for d in data_list]
            all_data.power_draws = [_ValueWithRank(d["power"], d["rank"], "W") for d in data_list]
            all_data.utilizations = [_ValueWithRank(d["util"], d["rank"], "%") for d in data_list]
            all_data.clock_rates = [_ValueWithRank(d["clock"], d["rank"], "MHz") for d in data_list]
            all_data.estimated_flops = [_ValueWithRank(d["flops"], d["rank"]) for d in data_list]

            return all_data

        # Wait for everyone here
        torch.distributed.barrier()

        return all_data


    def report_all_values(self, total_flops: float = 0.0, log_int):
        # Implement the modified report_all_values method here
        """Function to log metrics for all ranks over a time period

        It logs the metrics for all ranks, including min/max, over a specified time period.

        Args:
            total_flops (float, optional): The theoretical flops over the period. Defaults to 0.0.
            log_interval (int, optional): The training interval over which reporting is called (ms).
                                        Defaults to 0.

        Returns:
            bool: True if reported, else False
        """
        ret = False
        if not self._off and total_flops > 0.0 and log_interval > 0:
            elapsed, btime_us, temp, power, util, clock = self.elapsed()  # get raw time
            ptime = elapsed / (log_interval * 1.0)  # avg per iteration elapsed time, ms
            btime = btime_us / (log_interval * 1.0)  # avg per iteration get_batch time, us
            api_flops = total_flops / (log_interval * 1.0)  # avg per iteration flops, ms
            apir_flops = api_flops / (
                ptime * 10 ** 9 * self.world
            )  # this is avg per iteration this rank's thruput, TFLOP/s (note 10**9),
            et_flops = apir_flops / self.amp  # Estimated TFLOPs, not tracing backward

            o_dt = self._min_max(
                ptime, btime, float(temp), float(power), float(util), float(clock), et_flops,
            )
            if self.rank == 0:
                for rank in range(self.world):
                    now = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                    rank_data = o_dt.aflops[rank]()
                    logger.info(
                        f"{now} | Rank {rank}: "
                        f"Elapsed Time: {rank_data[0]} ms | "
                        f"Batch Time: {rank_data[1]} us | "
                        f"Temperature: {rank_data[2]} C | "
                        f"Power: {rank_data[3]} W | "
                        f"Utilization: {rank_data[4]} % | "
                        f"Clock: {rank_data[5]} MHz | "
                        f"Estimated Throughput: {rank_data[6]} TF/s"
                    )
                ret = True

        # Check/Communicate if tracking is turned off or on
        self._check_toggle()
        return ret




class globalStragglerDetector(Callback):
     def __init__(
        self,
    ) -> None:
   

    def init(self, state: State, logger: Logger) -> None:
        world_size = dist.get_world_size()
        stimer = StragglerDetector()
        port = int(os.environ.get('MASTER_PORT'))
        for rank in range(world_size):
            if rank == 0:
                stimer.configure(world_size, rank, enabled=True, port=port)
            else:
                stimer.configure(world_size, rank, enabled=True)
        

    def after_train_batch(self, state: State, logger: Logger):
        memory_report = {}

        model_device = next(state.model.parameters()).device
        if model_device.type != 'cuda':
            return

        memory_report = _get_memory_report(self.memory_keys)
        if self.dist_aggregate_batch_interval is not None and state.timestamp.batch.value % self.dist_aggregate_batch_interval == 0:
            dist_memory_report = {}
            for (mem_stat, val) in memory_report.items():
                dist_memory_report[mem_stat + '_avg'] = reduce_value(val, model_device, 'avg')
                dist_memory_report[mem_stat + '_min'] = reduce_value(val, model_device, 'min')
                dist_memory_report[mem_stat + '_max'] = reduce_value(val, model_device, 'max')
            memory_report.update(dist_memory_report)

        logger.log_metrics({f'memory/{mem_stat}': val for (mem_stat, val) in memory_report.items()})



gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ports = [65535, 65534, 65533, 65532, 65531, 65530, 65529, 65528]

detectors = []  # Create an empty list to store instances of StragglerDetector

for gpu_id, port in zip(gpu_ids, ports):
    detector = StragglerDetector()
    detector.configure(world=8, rank=gpu_id, port=port)
    detectors.append(detector)  # Store the detector instance in the list


# Arguments to configure 
#     world   : World Size
#     rank    : The rank of this trainer
#     mmcnt   : (Optional) Number of ranks to print for showing Min/Max Etpt
#     amp     : (Optional) Set to 3.0 if we only use timers in fwd pass
#     port    : (Optional) control port, useful only for rank-0
#     prefill : (Optional) howmany Events to pre-populate
#     enabled : (Optional) whether or not collection is enabled on startup


"""


