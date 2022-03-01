# Copyright 2021 MosaicML. All Rights Reserved.

import datetime
import logging
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
import warnings
from argparse import ArgumentParser
from typing import Any, List, Optional, Set

CLEANUP_TIMEOUT = datetime.timedelta(seconds=30)

log = logging.getLogger(__name__)


def get_parser():
    parser = ArgumentParser(description="Utility for launching distributed machine learning jobs.")

    parser.add_argument("-n",
                        "--nproc",
                        type=int,
                        required=True,
                        help="The number of processes to launch on this node. Required.")
    parser.add_argument("--world_size",
                        type=int,
                        default=-1,
                        help="The total number of processes to launch on all "
                        "nodes. Set to -1 to default to nproc (single-node operation). "
                        "Defaults to -1.")
    parser.add_argument("--base_rank",
                        type=int,
                        default=0,
                        help="The rank of the lowest ranked process to launch on this node. "
                        "Specifying a base_rank B and an nproc N will spawn processes "
                        "with global ranks [B, B+1, ... B+N-1]. Defaults to 0 (single-node "
                        "operation).")
    parser.add_argument("--node_rank",
                        type=int,
                        default=-1,
                        help="The rank of this node. Set to -1 to assume that all nodes have "
                        "the same number of processes, and calculate accordingly. Defaults to -1.")
    parser.add_argument("--master_addr",
                        type=str,
                        default="127.0.0.1",
                        help="The FQDN of the node running the rank 0 worker. Defaults to "
                        "127.0.0.1 (single-node operation).")
    parser.add_argument("--master_port",
                        type=int,
                        default=None,
                        help="The port on the master hosting the C10d TCP store. If you are running "
                        "multiple trainers on a single node, this generally needs to be unique for "
                        "each one. Defaults to a random free port.")
    parser.add_argument("--run_directory",
                        type=str,
                        default=None,
                        help=textwrap.dedent("""\
                            Directory to store run artifcats. 
                            Defaults to runs/{timestamp}/""")),
    parser.add_argument("-m",
                        "--module_mode",
                        action="store_true",
                        help="If set, run the training script as a module instead of as a script.")
    parser.add_argument("-v", "--verbose", action="store_true", help="If set, print verbose messages")
    parser.add_argument("training_script",
                        type=str,
                        help="The path to the training script used to initialize a single training "
                        "process. Should be followed by any command-line arguments the script "
                        "should be launched with.")
    parser.add_argument("training_script_args",
                        nargs="...",
                        help="Any arguments for the training script, given in the expected order.")

    return parser


def get_free_tcp_port() -> int:
    # from https://www.programcreek.com/python/?CodeExample=get+free+port
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port


def parse_args():
    parser = get_parser()

    args = parser.parse_args()
    if args.world_size == -1:
        args.world_size = args.nproc

    if args.node_rank == -1:
        if args.base_rank % args.nproc != 0:
            raise ValueError("node_rank not specified, but unable to infer since nodes appear to "
                             "have different amounts of processes.")
        args.node_rank = args.base_rank // args.nproc

    return args


def launch_processes(nproc: int, world_size: int, base_rank: int, node_rank: int, master_addr: str,
                     master_port: Optional[int], module_mode: bool, run_directory: Optional[str], training_script: str,
                     training_script_args: List[Any]) -> Set[subprocess.Popen]:
    log.info("Starting DDP on local node for global_rank(%s-%s)", base_rank, base_rank + nproc - 1)
    processes = []

    if run_directory is None:
        run_directory = os.path.join("runs", datetime.datetime.now().isoformat().replace(":", "-"))
    os.makedirs(run_directory, exist_ok=True)

    if master_port is None:
        warnings.warn("AutoSelectPortWarning: The DDP port was auto-selected. "
                      "This may lead to race conditions when launching multiple training processes simultaneously. "
                      "To eliminate this race condition, explicitely specify a port with --master_port PORT_NUMBER")
        master_port = get_free_tcp_port()
    log.info("DDP Store: tcp://%s:%s", master_addr, master_port)

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
        current_env["COMPOSER_RUN_DIRECTORY"] = run_directory

        log.info("Launching process for local_rank(%s), global_rank(%s) with command(%s)", local_rank, global_rank, cmd)

        if local_rank == 0:
            process = subprocess.Popen(cmd, env=current_env, text=True, shell=True)
        else:
            logs_dir = os.path.join(run_directory, f"rank_{global_rank}", "logs")
            os.makedirs(logs_dir, exist_ok=True)
            process = subprocess.Popen(
                cmd,
                # Using a shell to execute the command, so the env variables will be available to the CLI arguments
                shell=True,
                env=current_env,
                stdout=open(os.path.join(logs_dir, f"rank_{global_rank}.stdout.txt"), "x"),
                stderr=open(os.path.join(logs_dir, f"rank_{global_rank}.stderr.txt"), "x"),
                text=True,
            )
        processes.append(process)

    return set(processes)


def monitor_processes(processes: Set[subprocess.Popen]):
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


def print_process_exit_status(process: subprocess.Popen):
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


def cleanup_processes(processes: Set[subprocess.Popen]):
    living_processes_at_end = set()
    for process in processes:
        process.poll()
        if process.returncode is None:
            living_processes_at_end.add(process)
            log.info("Killing subprocess %s with SIGTERM", process.pid)
            try:
                os.killpg(process.pid, signal.SIGTERM)
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
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        process.poll()
        if process.returncode != 0 and process not in living_processes_at_end:
            # only print the processes that have actually crashed,
            # not the ones we killed
            print_process_exit_status(process)


def aggregate_process_returncode(processes: Set[subprocess.Popen]) -> int:
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
    args = parse_args()

    logging.basicConfig()
    log.setLevel(logging.INFO if args.verbose else logging.WARN)

    processes = launch_processes(nproc=args.nproc,
                                 world_size=args.world_size,
                                 base_rank=args.base_rank,
                                 node_rank=args.node_rank,
                                 master_addr=args.master_addr,
                                 master_port=args.master_port,
                                 module_mode=args.module_mode,
                                 training_script=args.training_script,
                                 run_directory=args.run_directory,
                                 training_script_args=args.training_script_args)

    try:
        monitor_processes(processes)
    except KeyboardInterrupt:
        print("Caught Ctrl+C; killing training processes")
        raise
    finally:
        cleanup_processes(processes)
        return aggregate_process_returncode(processes)


if __name__ == '__main__':
    sys.exit(main())
