# Copyright 2021 MosaicML. All Rights Reserved.

import datetime
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from typing import Any, List, Optional, Set

CLEANUP_TIMEOUT = datetime.timedelta(seconds=30)

log = logging.getLogger(__name__)


def get_parser():
    parser = ArgumentParser(description="Utility for launching distributed machine learning jobs.")

    required_args = parser.add_argument_group("required arguments")

    required_args.add_argument("-n",
                               "--nproc",
                               type=int,
                               help="The number of processes to launch on this node. Overrides env var "
                               "LOCAL_WORLD_SIZE.")

    parser.add_argument(
        "--stdout",
        type=str,
        default=None,
        help=("Format string for a filename to dump the STDOUT from the non-local-rank-zero processes. "
              "The local rank zero process will be piped through to STDOUT. The available format variables are: "
              "'{rank}', '{local_rank}', '{world_size}', '{node_rank}', and '{local_world_size}'. If specified, "
              "it is recommended to include '{rank}' or '{local_rank}' in the filename so each rank will write to its "
              "own file. By default, the STDOUT of the non-local-rank-zero processes is discarded; instead, use the "
              "FileLogger within Composer. This logger captures and saves the STDOUT of each process."),
    )
    parser.add_argument(
        "--stderr",
        type=str,
        default=None,
        help=("Format string for a filename to dump the STDERR from the non-local-rank-zero processes. "
              "The local rank zero process will be piped through to STDERR. The available format variables are: "
              "'{rank}', '{local_rank}', '{world_size}', '{node_rank}', and '{local_world_size}'. If specified, "
              "it is recommended to include '{rank}' or '{local_rank}' in the filename so each rank will write to its "
              "own file. By default, the STDERR of the non-local-rank-zero processes is discarded; instead, use the "
              "FileLogger within Composer. This logger captures and saves the STDERR of each process."),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="If set, print verbose messages")
    parser.add_argument("-m",
                        "--module_mode",
                        action="store_true",
                        help="If set, run the training script as a module instead of as a script.")

    multinode_args = parser.add_argument_group(
        "multi-node arguments",
        description="These arguments generally only need to be set when training in a multi-node "
        "environment, i.e. when the world_size is bigger than nproc.")
    multinode_args.add_argument("--world_size",
                                type=int,
                                help="The total number of processes to launch across all nodes. "
                                "Setting this to a value greater than nproc indicates a multi-node "
                                "environment. Overrides env var WORLD_SIZE. Defaults to nproc.")
    multinode_args.add_argument("--base_rank",
                                type=int,
                                help="The rank of the lowest ranked process to launch on this node. "
                                "Specifying a base_rank B and an nproc N will spawn processes with "
                                "global ranks [B, B+1, ... B+N-1]. In a multi-node environment, "
                                "at least one of base_rank and node_rank must be specified. "
                                "If only one of base_rank and node_rank are provided, it is assumed "
                                "that all nodes have the same amount of processes, and that the two "
                                "values are related as node_rank * nproc = base_rank. If this is "
                                "not the case, both base_rank and node_rank must be provided. "
                                "Overrides env var BASE_RANK. Defaults to 0 in a single-node "
                                "environment.")
    multinode_args.add_argument("--node_rank",
                                type=int,
                                help="The rank of this node. See base_rank for information on when "
                                "this must be provided. Overrides env var NODE_RANK. Defaults to 0 "
                                "in a single-node environment.")
    multinode_args.add_argument("--master_addr",
                                type=str,
                                help="The FQDN of the node hosting the C10d TCP store. For single-node "
                                "operation, this can generally be left as 127.0.0.1. Overrides env var "
                                "MASTER_ADDR. Defaults to 127.0.0.1 in a single-node environment.")
    multinode_args.add_argument("--master_port",
                                type=int,
                                help="The port on the master hosting the C10d TCP store. If you are "
                                "running multiple trainers on a single node, this generally needs "
                                "to be unique for each one. Overrides env var MASTER_PORT. Defaults "
                                "to a random free port in a single-node environment.")

    required_args.add_argument("training_script",
                               type=str,
                               help="The path to the training script used to initialize a single training "
                               "process. Should be followed by any command-line arguments the script "
                               "should be launched with.")
    required_args.add_argument("training_script_args",
                               nargs="...",
                               help="Any arguments for the training script, given in the expected order.")

    return parser


def _get_free_tcp_port() -> int:
    # from https://www.programcreek.com/python/?CodeExample=get+free+port
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port


def _parse_args():
    parser = get_parser()

    args = parser.parse_args()

    # Default values to env vars if they are not provided
    if args.nproc is None and "LOCAL_WORLD_SIZE" in os.environ:
        args.nproc = int(os.environ["LOCAL_WORLD_SIZE"])

    if args.world_size is None and "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.base_rank is None and "BASE_RANK" in os.environ:
        args.base_rank = int(os.environ["BASE_RANK"])

    if args.node_rank is None and "NODE_RANK" in os.environ:
        args.node_rank = int(os.environ["NODE_RANK"])

    if args.master_addr is None and "MASTER_ADDR" in os.environ:
        args.master_addr = os.environ["MASTER_ADDR"]

    if args.master_port is None and "MASTER_PORT" in os.environ:
        args.master_port = int(os.environ["MASTER_PORT"])

    if args.world_size is None:
        args.world_size = args.nproc

    if args.world_size < args.nproc:
        raise ValueError(f"world_size({args.world_size}) cannot be less than nproc({args.nproc})")

    is_multinode = args.world_size > args.nproc

    if is_multinode:
        if args.base_rank is None and args.node_rank is None:
            raise ValueError(f"In a multi-node environment, at least one of node_rank and base_rank must be provided.")

        if args.node_rank is None:
            if args.world_size % args.nproc != 0 or args.base_rank % args.nproc != 0:
                raise ValueError("node_rank not provided, but unable to infer from base_rank since nodes appear to "
                                 "have different amounts of processes. Please also specify node_rank.")
            args.node_rank = args.base_rank // args.nproc

        if args.base_rank is None:
            if args.world_size % args.nproc != 0:
                raise ValueError("base_rank not provided, but unable to infer from node_rank since nodes appear to "
                                 "have different amounts of processes. Please also provide base_rank.")
            args.base_rank = args.node_rank * args.nproc

        if args.base_rank + args.nproc >= args.world_size:
            raise ValueError(f"Cannot initialize processes for node with base_rank({args.base_rank}) and "
                             f"nproc({args.nproc}) because this would mean creating a process with "
                             f"rank({args.base_rank + args.nproc}), and all processes must have smaller rank than the "
                             f"world_size({args.world_size}).")

        if args.master_addr is None:
            raise ValueError("In a multi-node environment, master_addr is required.")

        if args.master_port is None:
            raise ValueError("In a multi-node environment, master_port is required.")

    else:
        if args.base_rank is not None and args.base_rank != 0:
            raise ValueError(f"base_rank({args.base_rank}) != 0 is not valid in a single-node environment.")
        args.base_rank = 0

        if args.node_rank is not None and args.node_rank != 0:
            raise ValueError(f"node_rank({args.node_rank}) != 0 is not valid in a single-node environment.")
        args.node_rank = 0

        if args.master_addr is None:
            args.master_addr = "127.0.0.1"

        if args.master_port is None:
            warnings.warn("AutoSelectPortWarning: The distributed key-value port was auto-selected. "
                          "This may lead to race conditions when launching multiple training processes simultaneously. "
                          "To eliminate this race condition, explicitly specify a port with --master_port PORT_NUMBER")
            args.master_port = _get_free_tcp_port()

    return args


def _launch_processes(nproc: int, world_size: int, base_rank: int, node_rank: int, master_addr: str, master_port: int,
                      module_mode: bool, training_script: str, stdout_file_format: Optional[str],
                      stderr_file_format: Optional[str], training_script_args: List[Any],
                      processes: Set[subprocess.Popen]):
    log.info("Starting distributed environment on local node for global_rank(%s-%s)", base_rank, base_rank + nproc - 1)
    log.info("Distributed KV store: tcp://%s:%s", master_addr, master_port)

    for local_rank in range(nproc):
        global_rank = base_rank + local_rank
        cmd = f"{sys.executable} -u"
        if module_mode:
            cmd += " -m"
        training_script_args_quoted = [f'"{arg}"' for arg in training_script_args]

        cmd += f" {training_script} {' '.join(training_script_args_quoted)}"

        current_env = os.environ.copy()
        current_env["RANK"] = str(global_rank)
        current_env["WORLD_SIZE"] = str(world_size)
        current_env["LOCAL_RANK"] = str(local_rank)
        current_env["LOCAL_WORLD_SIZE"] = str(nproc)
        current_env["NODE_RANK"] = str(node_rank)
        current_env["MASTER_ADDR"] = master_addr
        current_env["MASTER_PORT"] = str(master_port)

        log.info("Launching process for local_rank(%s), global_rank(%s) with command(%s)", local_rank, global_rank, cmd)

        if local_rank == 0:
            process = subprocess.Popen(cmd, env=current_env, text=True, shell=True)
        else:

            def _get_file(format: Optional[str]):
                if format is None:
                    return subprocess.DEVNULL
                else:
                    filename = format.format(
                        rank=global_rank,
                        world_size=world_size,
                        local_rank=local_rank,
                        local_world_size=nproc,
                        node_rank=node_rank,
                    )
                    return open(filename, 'x')

            process = subprocess.Popen(
                cmd,
                # Using a shell to execute the command, so the env variables will be available to the CLI arguments
                shell=True,
                env=current_env,
                stdout=_get_file(stdout_file_format),
                stderr=_get_file(stderr_file_format),
                text=True,
            )
        processes.add(process)


def _monitor_processes(processes: Set[subprocess.Popen]):
    while len(processes) > 0:
        process_has_crashed = False
        for process in processes:
            if process.poll() is None:
                # the process is still running
                continue
            else:
                # return code of 0 implies clean exit
                if process.returncode != 0:
                    process_has_crashed = True
                    break
                else:
                    # exited cleanly
                    processes.remove(process)
                    break
        if process_has_crashed:
            break
        time.sleep(1)


def _print_process_exit_status(process: subprocess.Popen):
    if process.stdout is None:
        output = None
    else:
        output = process.stdout.read()

    if process.stderr is None:
        stderr = None
    else:
        stderr = process.stderr.read()
    exc = subprocess.CalledProcessError(
        process.returncode,
        cmd=process.args,
        output=output,
        stderr=stderr,
    )
    error_msg = [f"Process {process.pid} excited with code {process.returncode}"]
    if output is not None:
        error_msg.extend([
            "----------Begin subprocess STDOUT----------",
            output,
            "----------End subprocess STDOUT----------",
        ])
    if stderr is not None:
        error_msg.extend([
            "----------Begin subprocess STDERR----------",
            exc.stderr,
            "----------End subprocess STDERR----------",
        ])
    print("\n".join(error_msg))


def _cleanup_processes(processes: Set[subprocess.Popen]):
    living_processes_at_end = set()
    for process in processes:
        process.poll()
        if process.returncode is None:
            living_processes_at_end.add(process)
            log.info("Killing subprocess %s with SIGTERM", process.pid)
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    current_time = datetime.datetime.now()
    print(f"Waiting up to {CLEANUP_TIMEOUT.seconds} seconds for all training processes to terminate...")
    while datetime.datetime.now() - current_time < CLEANUP_TIMEOUT:
        for process in processes:
            process.poll()
        if all(process.returncode is not None for process in processes):
            break
        time.sleep(0.1)

    for process in processes:
        process.poll()
        if process.returncode is None:
            log.warning("Failed to kill subprocess %s with SIGTERM; using SIGKILL instead", process.pid)
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        process.poll()
        if process.returncode != 0 and process not in living_processes_at_end:
            # only print the processes that have actually crashed,
            # not the ones we killed
            _print_process_exit_status(process)


def _aggregate_process_returncode(processes: Set[subprocess.Popen]) -> int:
    for process in processes:
        process.poll()
        if process.returncode is None:
            log.error("Subprocess %s has still not exited; return exit code 1.", process.pid)
            return 1
        if process.returncode != 0:
            log.error("Subprocess %s exited with code %s", process.pid, process.returncode)
            return process.returncode

    return 0


def main():
    args = _parse_args()

    logging.basicConfig()
    log.setLevel(logging.INFO if args.verbose else logging.WARN)

    processes = set()

    try:
        _launch_processes(nproc=args.nproc,
                          world_size=args.world_size,
                          base_rank=args.base_rank,
                          node_rank=args.node_rank,
                          master_addr=args.master_addr,
                          master_port=args.master_port,
                          module_mode=args.module_mode,
                          stdout_file_format=args.stdout,
                          stderr_file_format=args.stderr,
                          training_script=args.training_script,
                          training_script_args=args.training_script_args,
                          processes=processes)
        _monitor_processes(processes)
    except:
        # Print the exception first, then kill the training processes, since killing
        # may take up to CLEANUP_TIMEOUT seconds, and the user should know immediately
        # what failed. No need to re-raise the exception, as `aggregate_process_returncode`
        # will return an appropriate error code, which will cause the script to exit.
        traceback.print_exc()
        print("Killing training processes")
    finally:
        _cleanup_processes(processes)
        return _aggregate_process_returncode(processes)


if __name__ == '__main__':
    sys.exit(main())
