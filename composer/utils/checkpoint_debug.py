from typing import Optional
import warnings

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner


from torch.distributed.checkpoint.storage import (
    StorageWriter,
)

from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.utils import _DistWrapper

import logging
log = logging.getLogger(__name__)

def save_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[SavePlanner] = None,
) -> Metadata:
    """This method is deprecated. Please switch to 'save'."""
    warnings.warn(
        "'save_state_dict' is deprecated and will be removed in future versions. Please use 'save' instead."
    )

    # TODO: test returning `save` here instead.
    return _save_state_dict(state_dict, storage_writer, process_group, coordinator_rank, no_dist, planner)

def _save_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[SavePlanner] = None,
) -> Metadata:
    log.warning('starting pytorch save state dict')

    torch._C._log_api_usage_once("torch.distributed.checkpoint.save_state_dict")

    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    distW.reduce_scatter = reduce_scatter.__get__(distW, _DistWrapper)
    if planner is None:
        planner = DefaultSavePlanner()
    assert planner is not None

    global_metatadata = None

    def local_step():
        log.warning('starting local step')
        assert planner is not None
        planner.set_up_planner(state_dict, distW.is_coordinator)
        storage_writer.set_up_storage_writer(distW.is_coordinator)
        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        log.warning('finished local step')
        return local_plan

    def global_step(all_local_plans):
        log.warning('starting global step')
        nonlocal global_metatadata

        assert planner is not None
        all_local_plans, global_metatadata = planner.create_global_plan(
            all_local_plans
        )
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        log.warning('finished global step')
        return all_local_plans

    log.warning('starting reduce scatter')
    central_plan = distW.reduce_scatter("plan", local_step, global_step)

    def write_data():
        log.warning('starting write data')
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_writes = storage_writer.write_data(final_local_plan, planner)

        all_writes.wait()
        log.warning('finished write data')
        return all_writes.value()

    def finish_checkpoint(all_results):
        log.warning('starting finish checkpoint')
        assert global_metatadata is not None
        storage_writer.finish(metadata=global_metatadata, results=all_results)
        log.warning('finished finish checkpoint')
        return global_metatadata

    log.warning('starting all reduce')
    return distW.all_reduce("write", write_data, finish_checkpoint)

from typing import (
    List,
    Callable,
    Optional,
    Union,
    TypeVar,
    cast,
)
from torch.distributed.checkpoint.api import (
    CheckpointException,
    _wrap_exception,
    WRAPPED_EXCEPTION,
)
from torch.distributed.checkpoint.utils import _get_failure_dict
T = TypeVar("T")
R = TypeVar("R")


def reduce_scatter(
    self,
    step: str,
    map_fun: Callable[[], T],
    reduce_fun: Callable[[List[T]], List[R]],
) -> R:
    """
    Compute a value on each rank, then do centralized reduce on a single rank, followed by a scatter.

    This method operates in the following way:
        Run ``map_fun`` on all ranks
        Gather results on rank 0
        Call ``reduce_fun`` on all those values
        Scatter to each rank part of the result.
    """
    local_data: Union[WRAPPED_EXCEPTION, T]
    try:
        local_data = map_fun()
    except BaseException as e:
        local_data = _wrap_exception(e)

    log.warning('starting gather')
    all_data = self.gather_object(local_data)
    log.warning('finished gather')
    all_results: Optional[List[Union[R, CheckpointException]]] = None
    log.warning('starting rank 0 work')
    if self.is_coordinator:
        assert all_data is not None
        node_failures = _get_failure_dict(all_data)

        if len(node_failures) == 0:
            try:
                # N.B. why can't mypy cast List[R] to List[Union[R, WRAPPED_EXCEPTION]]?
                all_results = cast(
                    List[Union[R, CheckpointException]],
                    reduce_fun(cast(List[T], all_data)),
                )
            except BaseException as e:
                node_failures[self.rank] = _wrap_exception(e)

        if len(node_failures) > 0:
            all_results = [CheckpointException(step, node_failures)] * self.get_world_size()
    log.warning('finished rank 0 work')

    log.warning('starting scatter')
    result = self.scatter_object(all_results)
    log.warning('finished scatter')
    if isinstance(result, CheckpointException):
        raise result
    return result
