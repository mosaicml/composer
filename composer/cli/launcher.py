# Copyright 2021 MosaicML. All Rights Reserved.

import datetime
import os
import signal
import subprocess
import sys
import tempfile
import time
from argparse import ArgumentParser
from typing import Any, List, Set

import torch.distributed

CLEANUP_TIMEOUT = datetime.timedelta(seconds=5)


def parse_args():
    parser = ArgumentParser(description="Utility for launching distributed jobs with composer.")

    parser.add_argument("-n",
                        "--nproc",
                        type=int,
                        required=True,
                        help="The number of processes to launch on this node.")
    parser.add_argument("--world_size",
                        type=int,
                        default=-1,
                        help="The total number of processes to launch on all"
                        "nodes. Set to -1 to default to nproc (single-node operation).")
    parser.add_argument("--base_rank",
                        type=int,
                        default=0,
                        help="The rank of the lowest ranked process to launch on this node."
                        "Specifying a base_rank B and an nproc N will spawn processes"
                        "with global ranks [B, B+1, ... B+N-1].")
    parser.add_argument("--master_addr",
                        type=str,
                        default="127.0.0.1",
                        help="The FQDN of the node running the rank 0 worker.")
    parser.add_argument("--master_port",
                        type=int,
                        default=29400,
                        help="The port on the master hosting the C10d TCP store.")
    parser.add_argument("-m",
                        "--module_mode",
                        action="store_true",
                        help="Run the training script as a module instead of as a script.")
    parser.add_argument("training_script",
                        type=str,
                        help="The path to the training script used to initialize a single training "
                        "process. Should be followed by any command-line arguments the script "
                        "should be launched with.")
    parser.add_argument("training_script_args", nargs="...")

    args = parser.parse_args()
    if args.world_size == -1:
        args.world_size = args.nproc

    return args


def launch_processes(nproc: int, world_size: int, base_rank: int, master_addr: str, master_port: int, module_mode: bool,
                     training_script: str, training_script_args: List[Any]) -> Set[subprocess.Popen]:
    print("Starting DDP on node_rank(%d) with world_size(%d)", 0, nproc)
    processes = []

    for local_rank in range(nproc):
        global_rank = base_rank + local_rank
        if module_mode:
            cmd = [sys.executable, '-u', '-m', training_script, *training_script_args]
        else:
            cmd = [sys.executable, '-u', training_script, *training_script_args]

        current_env = os.environ.copy()
        current_env["RANK"] = str(global_rank)
        current_env["WORLD_SIZE"] = str(world_size)
        current_env["LOCAL_RANK"] = str(local_rank)
        current_env["LOCAL_WORLD_SIZE"] = str(nproc)
        current_env["MASTER_ADDR"] = master_addr
        current_env["MASTER_PORT"] = str(master_port)

        print("Launching process for local_rank(%d), global_rank(%d)", local_rank, global_rank)

        if local_rank == 0:
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
                    print("\n".join(error_msg))
                    print(exc)
                    sys.exit(process.returncode)
                else:
                    # exited cleanly
                    processes.remove(process)
                    break
        time.sleep(1)


def cleanup_processes(processes: Set[subprocess.Popen]):
    for process in processes:
        if process.returncode is None:
            print("Killing subprocess %s with SIGTERM", process.pid)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    current_time = datetime.datetime.now()
    print(f"Waiting {CLEANUP_TIMEOUT.seconds} seconds for processes to terminate...")
    while datetime.datetime.now() - current_time < CLEANUP_TIMEOUT:
        if all(process.returncode is not None for process in processes):
            break
        time.sleep(0.1)

    for process in processes:
        if process.returncode is None:
            print("Failed to kill subprocess %s with SIGTERM; using SIGKILL instead", process.pid)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main():
    args = parse_args()

    processes = launch_processes(nproc=args.nproc,
                                 world_size=args.world_size,
                                 base_rank=args.base_rank,
                                 master_addr=args.master_addr,
                                 master_port=args.master_port,
                                 module_mode=args.module_mode,
                                 training_script=args.training_script,
                                 training_script_args=args.training_script_args)

    try:
        monitor_processes(processes)
    finally:
        cleanup_processes(processes)
        return 0


if __name__ == '__main__':
    sys.exit(main())
