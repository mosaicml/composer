import json
import logging
import os
import platform
import sys
import warnings
from typing import Dict, Optional

import cpuinfo
import psutil
import torch
from mlperf_logging import mllog

import composer
from composer import Callback, State
from composer.loggers import Logger
from composer.utils import dist

try:
    from mlperf_logging import mllog
    from mlperf_logging.mllog import constants
    mlperf_available = True
except ImportError:
    mlperf_available = False

BENCHMARKS = ("resnet")
DIVISIONS = ("open")
STATUS = ("onprem", "cloud", "preview")


def rank_zero() -> bool:
    return dist.get_global_rank() == 0


class MLPerfCallback(Callback):
    """Creates a compliant results file for MLPerf Training benchmark.

    A submission folder structure will be created with the ``root_folder``
    as the base and the following directories:

    root_folder/
        results/
            [system_name]/
                [benchmark]/
                    results_0.txt
                    results_1.txt
                    ...
        systems/
            [system_name].json

    A required systems description will be automatically generated,
    and best effort made to populate the fields, but should be manually
    checked prior to submission.

    Currently, only OPEN division submissions are supported with this Callback.

    Args:
        root_folder (str): The root submission folder
        index (int): The repetition index of this run. The filename created will be
            ``result_[index].txt``.
        submitter (str, optional): Submitting organization. Default: MosaicML.
        system_name (str, optional): Name of the system (e.g. 8xA100_composer). If
            not provided, system name will default to ``[world_size]x[device_name]_composer``,
            e.g. ``8xNVIDIA_A100_80GB_composer.
        benchmark (str, optional): Benchmark name. Default: ``"resnet"``.
        division (str, optional): Division of submission. Currently only open division is
            supported. Default: ``"open"``.
        status (str, optional): Submission status. One of (onprem, cloud, or preview).
            Default: ``"onprem"``.
        target (float, optional): The target metric before the mllogger marks the stop
            of the timing run. Default: ``0.759`` (resnet benchmark).
    """

    def __init__(
        self,
        root_folder: str,
        index: int,
        submitter: str = "MosaicML",
        system_name: Optional[str] = None,
        benchmark: str = "resnet",
        division: str = "open",
        status: str = "onprem",
        target: float = 0.759,
    ) -> None:

        if benchmark not in BENCHMARKS:
            raise ValueError(f"benchmark: {benchmark} must be one of {BENCHMARKS}")
        if division not in DIVISIONS:
            raise ValueError(f"division: {division} must be one of {DIVISIONS}")
        if status not in STATUS:
            raise ValueError(f"status: {status} must be one of {STATUS}")
        if not mlperf_available:
            raise ValueError("MLperf logger is required")
        self.mllogger = mllog.get_mllogger()
        self.target = target
        self.system_name = system_name
        self.benchmark = benchmark
        self.root_folder = root_folder

        system_desc = get_system_description(submitter, division, status, system_name)
        system_name = system_desc['system_name']

        self._create_submission_folders(root_folder, system_name, benchmark)

        # save system description file
        systems_path = os.path.join(root_folder, 'systems', f'{system_name}.json')
        if os.path.exists(systems_path):
            with open(systems_path, 'r') as f:
                existing_systems_desc = json.load(f)
                if sorted(existing_systems_desc.items()) != sorted(system_desc.items()):
                    raise ValueError(f'Existing system description in {systems_path} does not match this machine.')
        else:
            with open(systems_path, 'w') as f:
                json.dump(system_desc, f, indent=4)

        filename = os.path.join(root_folder, 'results', system_name, benchmark, f'result_{index}.txt')
        if os.path.exists(filename):
            raise FileExistsError(f'{filename} already exists.')

        self._file_handler = logging.FileHandler(filename)
        self._file_handler.setLevel(logging.INFO)
        self.mllogger.logger.addHandler(self._file_handler)

        # TODO: implement cache clearing
        self.mllogger.start(key=mllog.constants.CACHE_CLEAR)
        self.mllogger.start(key=mllog.constants.INIT_START)

        if rank_zero():
            self._log_dict({
                constants.SUBMISSION_BENCHMARK: benchmark,
                constants.SUBMISSION_DIVISION: division,
                constants.SUBMISSION_ORG: submitter,
                constants.SUBMISSION_PLATFORM: system_name,
                constants.SUBMISSION_STATUS: status,
            })

    def _create_submission_folders(self, root_folder: str, system_name: str, benchmark: str):
        if not os.path.isdir(root_folder):
            raise FileNotFoundError(f"{root_folder} not found.")

        results_folder = os.path.join(root_folder, 'results')
        log_folder = os.path.join(root_folder, 'results', system_name)
        benchmark_folder = os.path.join(log_folder, benchmark)
        systems_folder = os.path.join(root_folder, 'systems')

        os.makedirs(results_folder, exist_ok=True)
        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(benchmark_folder, exist_ok=True)
        os.makedirs(systems_folder, exist_ok=True)

    def _log_dict(self, data: Dict):
        for key, value in data.items():
            self.mllogger.event(key=key, value=value)

    def fit_start(self, state: State, logger: Logger) -> None:
        if rank_zero():
            if state.train_dataloader.batch_size is None:
                raise ValueError("Batch size is required to be set for dataloader.")

            self._log_dict({
                constants.SEED: state.seed,
                constants.GLOBAL_BATCH_SIZE: state.train_dataloader.batch_size * dist.get_world_size(),
                constants.GRADIENT_ACCUMULATION_STEPS: state.grad_accum,
                constants.TRAIN_SAMPLES: len(state.train_dataloader.dataset),
                constants.EVAL_SAMPLES: len(state.evaluators[0].dataloader.dataloader.dataset)
            })

        self.mllogger.event(key=constants.INIT_STOP)

        dist.barrier()
        if rank_zero():
            self.mllogger.event(key=constants.RUN_START)

    def epoch_start(self, state: State, logger: Logger) -> None:
        if rank_zero():
            self.mllogger.event(key=constants.EPOCH_START, metadata={'epoch_num': state.timer.epoch.value})
            self.mllogger.event(key=constants.BLOCK_START,
                                metadata={
                                    'first_epoch_num': state.timer.epoch.value,
                                    'epoch_count': 1
                                })

    def epoch_end(self, state: State, logger: Logger) -> None:
        if rank_zero():
            self.mllogger.event(key=constants.EPOCH_STOP, metadata={'epoch_num': state.timer.epoch.value})

    def eval_start(self, state: State, logger: Logger) -> None:
        if rank_zero():
            self.mllogger.event(key=constants.EVAL_START, metadata={'epoch_num': state.timer.epoch.value})

    def eval_end(self, state: State, logger: Logger) -> None:
        if rank_zero():
            accuracy = 0.99  # TODO: retrieve accuracy from metrics

            self.mllogger.event(key=constants.EVAL_STOP, metadata={'epoch_num': state.timer.epoch.value})
            self.mllogger.event(key=constants.EVAL_ACCURACY,
                                value=accuracy,
                                metadata={'epoch_num': state.timer.epoch.value})
            self.mllogger.event(key=constants.BLOCK_STOP, metadata={'first_epoch_num': state.timer.epoch.value})

            if accuracy > self.target:
                self.mllogger.event(key=constants.RUN_STOP, metadata={"status": "success"})
                self.mllogger.logger.removeHandler(self._file_handler)


def get_system_description(
    submitter: str,
    division: str,
    status: str,
    system_name: Optional[str] = None,
) -> Dict[str, str]:
    """Generates a valid system description.

    Make a best effort to auto-populate some of the fields, but should
    be manually checked prior to submission. The system name is
    auto-generated as "[world_size]x[device_name]_composer", e.g.
    "8xNVIDIA_A100_80GB_composer".

    Args:
        submitter (str): Name of the submitter organization
        division (str): Submission division (open, closed)
        status (str): system status (cloud, onprem, preview)

    Returns:
        system description as a dictionary
    """
    is_cuda = torch.cuda.is_available()
    cpu_info = cpuinfo.get_cpu_info()

    system_desc = {
        "submitter": submitter,
        "division": division,
        "status": status,
        "number_of_nodes": dist.get_world_size() / dist.get_local_world_size(),
        "host_processors_per_node": "",
        "host_processor_model_name": str(cpu_info.get('brand_raw', "CPU")),
        "host_processor_core_count": str(psutil.cpu_count(logical=False)),
        "host_processor_vcpu_count": "",
        "host_processor_frequency": cpu_info.get('hz_advertised_friendly', ""),
        "host_processor_caches": "",
        "host_processor_interconnect": "",
        "host_memory_capacity": "",
        "host_storage_type": "",
        "host_storage_capacity": "",
        "host_networking": "",
        "host_networking_topology": "",
        "host_memory_configuration": "",
        "accelerators_per_node": str(dist.get_local_world_size()) if is_cuda else "0",
        "accelerator_model_name": str(torch.cuda.get_device_name(None)) if is_cuda else "",
        "accelerator_host_interconnect": "",
        "accelerator_frequency": "",
        "accelerator_on-chip_memories": "",
        "accelerator_memory_configuration": "",
        "accelerator_memory_capacity": "",
        "accelerator_interconnect": "",
        "accelerator_interconnect_topology": "",
        "cooling": "",
        "hw_notes": "",
        "framework": f"PyTorch v{torch.__version__} and MosaicML composer v{composer.__version__}",
        "other_software_stack": {
            "cuda_version": torch.version.cuda if is_cuda else "",
            "composer_version": composer.__version__,
            "python_version": sys.version,
        },
        "operating_system": f"{platform.system()} {platform.release()}",
        "sw_notes": "",
    }

    if system_desc['number_of_nodes'] != 1:
        warnings.warn("Number of nodes > 1 not tested, proceed with caution.")

    if system_name is None:
        world_size = dist.get_world_size()
        if is_cuda:
            device_name = system_desc['accelerator_model_name']
        else:
            device_name = system_desc['host_processor_model_name']

        device_name = device_name.replace(' ', '_')
        system_name = f"{world_size}x{device_name}_composer"

    # default to system name as "[world_size]x[device_name]"
    # e.g. 8xNVIDIA_A100_80GB
    system_desc['system_name'] = system_name
    return system_desc