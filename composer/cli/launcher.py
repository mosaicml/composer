# Copyright 2021 MosaicML. All Rights Reserved.

import datetime
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from argparse import ArgumentParser
from typing import Any, List, Set

logger = logging.getLogger(__name__)

import torch.distributed


def parse_args():
    parser = ArgumentParser(description="Utility for launching distributed jobs with composer.")

    parser.add_argument("-n", "--nproc", type=int, required=True, help="The number of process to launch on this node.")
    parser.add_argument("training_script",
                        type=str,
                        help="The path to the training script used to initialize a single training "
                        "process. Should be followed by any command-line arguments the script "
                        "should be launched with.")
    parser.add_argument("training_script_args", nargs="...")

    return parser.parse_args()


def launch_processes(nproc: int, training_script: str, training_script_args: List[Any]) -> Set[subprocess.Popen]:
    logger.info("Starting DDP on node_rank(%d) with world_size(%d)", 0, nproc)
    processes = []

    for rank in range(nproc):
        cmd = [sys.executable, training_script, *training_script_args]

        current_env = os.environ.copy()
        current_env["RANK"] = str(rank)
        current_env["WORLD_SIZE"] = str(nproc)
        current_env["LOCAL_RANK"] = str(rank)
        current_env["LOCAL_WORLD_SIZE"] = str(nproc)
        current_env["MASTER_ADDR"] = "127.0.0.1"
        current_env["MASTER_PORT"] = str(29400)

        logger.info("Launching process for global_rank(%d) on node_rank(%d)", rank, 0)

        if rank == 0:
            process = subprocess.Popen(cmd, env=current_env, text=True)
        else:
            process = subprocess.Popen(
                cmd,
                env=current_env,
                stdout=tempfile.TemporaryFile(),
                stderr=tempfile.TemporaryFile(),
                text=True,
            )
        processes.append(process)

    return set(processes)


def monitor_processes(processes: Set[subprocess.Popen]):
    while len(processes) > 0:
        for process in processes:
            if process.poll() is None:
                # the process is still running
                continue
            else:
                # return code of 0 implies clean exit
                # return code of -9 implies sigkill, presumably from cleanup_processes()
                if process.returncode not in (0, -9):
                    if process.stdout is None:
                        output = ""
                    else:
                        output = process.stdout.read()

                    if process.stderr is None:
                        stderr = ""
                    else:
                        stderr = process.stderr.read()
                    exc = subprocess.CalledProcessError(
                        process.returncode,
                        cmd=process.args,
                        output=output,
                        stderr=stderr,
                    )
                    error_msg = [
                        "Error in subprocess",
                        "----------Subprocess STDOUT----------",
                        exc.output,
                        "----------Subprocess STDERR----------",
                        exc.stderr,
                    ]
                    logger.exception("\n".join(error_msg), exc_info=exc)
                    sys.exit(process.returncode)
                else:
                    # exited cleanly
                    processes.remove(process)
                    break
        time.sleep(1)


def cleanup_processes(processes: Set[subprocess.Popen]):
    for process in processes:
        if process.returncode is None:
            logger.info("Killing subprocess %s with SIGTERM", process.pid)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    current_time = datetime.datetime.now()
    while datetime.datetime.now() - current_time < datetime.timedelta(seconds=5):
        if all(process.returncode is not None for process in processes):
            break
        time.sleep(0.1)

    for process in processes:
        if process.returncode is None:
            logger.error("Killing subprocess %s with SIGKILL", process.pid)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main():
    args = parse_args()

    processes = launch_processes(args.nproc, args.training_script, args.training_script_args)

    try:
        monitor_processes(processes)
    finally:
        cleanup_processes(processes)
