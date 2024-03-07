#!/usr/bin/env python3
# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The Composer CLI launcher for distributed training."""

import contextlib
import datetime
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from argparse import ArgumentParser
from typing import Any, Dict, List, Union

import psutil
import torch

import composer
from composer.loggers.mosaicml_logger import (
    MOSAICML_GPU_LOG_FILE_PREFIX_ENV_VAR,
    MOSAICML_LOG_DIR_ENV_VAR,
    MOSAICML_PLATFORM_ENV_VAR,
)
from composer.utils import get_free_tcp_port

CLEANUP_TIMEOUT = datetime.timedelta(seconds=30)

log = logging.getLogger(__name__)


def _get_parser():
    parser = ArgumentParser(description='Utility for launching distributed machine learning jobs.')

    parser.add_argument('--version', action='version', version=f'MosaicML Composer {composer.__version__}')

    required_args = parser.add_argument_group('required arguments')

    parser.add_argument(
        '-n',
        '--nproc',
        type=int,
        help=(
            'The number of processes to launch on this node. Overrides env var `LOCAL_WORLD_SIZE` if specified; '
            'otherwise, defaults to `max(1, torch.cuda.device_count())`.'
        ),
    )

    parser.add_argument(
        '--stdout',
        type=str,
        default=None,
        help=(
            'Format string for a filename to dump the STDOUT from the non-local-rank-zero processes. '
            'The local rank zero process will be piped through to STDOUT. The available format variables are: '
            "'{rank}', '{local_rank}', '{world_size}', '{node_rank}', and '{local_world_size}'. If specified, "
            "it is recommended to include '{rank}' or '{local_rank}' in the filename so each rank will write to its "
            'own file. By default, the STDOUT of the non-local-rank-zero processes is discarded; instead, use the '
            'FileLogger within Composer. This logger captures and saves the STDOUT of each process.'
        ),
    )
    parser.add_argument(
        '--stderr',
        type=str,
        default=None,
        help=(
            'Format string for a filename to dump the STDERR from the non-local-rank-zero processes. '
            'The local rank zero process will be piped through to STDERR. The available format variables are: '
            "'{rank}', '{local_rank}', '{world_size}', '{node_rank}', and '{local_world_size}'. If specified, "
            "it is recommended to include '{rank}' or '{local_rank}' in the filename so each rank will write to its "
            'own file. By default, the STDERR of the non-local-rank-zero processes is discarded; instead, use the '
            'FileLogger within Composer. This logger captures and saves the STDERR of each process.'
        ),
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='If set, print verbose messages')
    parser.add_argument(
        '-m',
        '--module_mode',
        action='store_true',
        help=(
            'If set, run the training script as a module instead of as a script. '
            'Cannot be used in conjunction with `command_mode`'
        ),
    )
    parser.add_argument(
        '-c',
        '--command_mode',
        action='store_true',
        help=(
            'If set, run the training script as a command (i.e. without `python`). '
            'Cannot be used in conjunction with `module_mode`.'
        ),
    )

    multinode_args = parser.add_argument_group(
        'multi-node arguments',
        description=(
            'These arguments generally only need to be set when training in a multi-node '
            'environment, i.e. when the world_size is bigger than nproc.'
        ),
    )
    multinode_args.add_argument(
        '--world_size',
        type=int,
        help=(
            'The total number of processes to launch across all nodes. '
            'Setting this to a value greater than nproc indicates a multi-node '
            'environment. Overrides env var WORLD_SIZE. Defaults to nproc.'
        ),
    )
    multinode_args.add_argument(
        '--base_rank',
        type=int,
        help=(
            'The rank of the lowest ranked process to launch on this node. '
            'Specifying a base_rank B and an nproc N will spawn processes with '
            'global ranks [B, B+1, ... B+N-1]. In a multi-node environment, '
            'at least one of base_rank and node_rank must be specified. '
            'If only one of base_rank and node_rank are provided, it is assumed '
            'that all nodes have the same amount of processes, and that the two '
            'values are related as node_rank * nproc = base_rank. If this is '
            'not the case, both base_rank and node_rank must be provided. '
            'Overrides env var BASE_RANK. Defaults to 0 in a single-node '
            'environment.'
        ),
    )
    multinode_args.add_argument(
        '--node_rank',
        type=int,
        help=(
            'The rank of this node. See base_rank for information on when '
            'this must be provided. Overrides env var NODE_RANK. Defaults to 0 '
            'in a single-node environment.'
        ),
    )
    multinode_args.add_argument(
        '--master_addr',
        type=str,
        help=(
            'The FQDN of the node hosting the C10d TCP store. For single-node '
            'operation, this can generally be left as 127.0.0.1. Overrides env var '
            'MASTER_ADDR. Defaults to 127.0.0.1 in a single-node environment.'
        ),
    )
    multinode_args.add_argument(
        '--master_port',
        type=int,
        help=(
            'The port on the master hosting the C10d TCP store. If you are '
            'running multiple trainers on a single node, this generally needs '
            'to be unique for each one. Overrides env var MASTER_PORT. Defaults '
            'to a random free port in a single-node environment.'
        ),
    )

    required_args.add_argument(
        'training_script',
        type=str,
        help=(
            'The path to the training script used to initialize a single training '
            'process. Should be followed by any command-line arguments the script '
            'should be launched with.'
        ),
    )
    required_args.add_argument(
        'training_script_args',
        nargs='...',
        help='Any arguments for the training script, given in the expected order.',
    )

    return parser


def _parse_args():
    parser = _get_parser()

    args = parser.parse_args()

    # Default values to env vars if they are not provided
    if args.nproc is None:
        if 'LOCAL_WORLD_SIZE' in os.environ:
            args.nproc = int(os.environ['LOCAL_WORLD_SIZE'])
        else:
            args.nproc = torch.cuda.device_count()

        if args.nproc == 0:
            # This could happen if doing cpu-only training,
            # which could cause torch.cuda.device_count() to return 0,
            # and LOCAL_WORLD_SIZE (as set by MCLI) to be zero
            args.nproc = 1

    if args.nproc < 1:
        raise ValueError('The nproc must be 1 or greater')

    if args.world_size is None and 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])

    if args.base_rank is None and 'BASE_RANK' in os.environ:
        args.base_rank = int(os.environ['BASE_RANK'])

    if args.node_rank is None and 'NODE_RANK' in os.environ:
        args.node_rank = int(os.environ['NODE_RANK'])

    if args.master_addr is None and 'MASTER_ADDR' in os.environ:
        args.master_addr = os.environ['MASTER_ADDR']

    if args.master_port is None and 'MASTER_PORT' in os.environ:
        args.master_port = int(os.environ['MASTER_PORT'])

    if args.world_size is None:
        args.world_size = args.nproc

    if args.world_size < args.nproc:
        raise ValueError(f'world_size({args.world_size}) cannot be less than nproc({args.nproc})')

    if args.world_size < 1:
        raise ValueError('The world_size must be 1 or greater')

    is_multinode = args.world_size > args.nproc

    if is_multinode:
        if args.base_rank is None and args.node_rank is None:
            raise ValueError(f'In a multi-node environment, at least one of node_rank and base_rank must be provided.')

        if args.node_rank is None:
            if args.world_size % args.nproc != 0 or args.base_rank % args.nproc != 0:
                raise ValueError(
                    'node_rank not provided, but unable to infer from base_rank since nodes appear to '
                    'have different amounts of processes. Please also specify node_rank.',
                )
            args.node_rank = args.base_rank // args.nproc

        if args.base_rank is None:
            if args.world_size % args.nproc != 0:
                raise ValueError(
                    'base_rank not provided, but unable to infer from node_rank since nodes appear to '
                    'have different amounts of processes. Please also provide base_rank.',
                )
            args.base_rank = args.node_rank * args.nproc

        if args.base_rank + args.nproc > args.world_size:
            raise ValueError(
                f'Cannot initialize processes for node with base_rank({args.base_rank}) and '
                f'nproc({args.nproc}) because this would mean creating a process with '
                f'rank({args.base_rank + args.nproc - 1}), and all processes must have smaller rank than '
                f'the world_size({args.world_size}).',
            )

        if args.master_addr is None:
            raise ValueError('In a multi-node environment, master_addr is required.')

        if args.master_port is None:
            raise ValueError('In a multi-node environment, master_port is required.')

    else:
        if args.base_rank is not None and args.base_rank != 0:
            raise ValueError(f'base_rank({args.base_rank}) != 0 is not valid in a single-node environment.')
        args.base_rank = 0

        if args.node_rank is not None and args.node_rank != 0:
            raise ValueError(f'node_rank({args.node_rank}) != 0 is not valid in a single-node environment.')
        args.node_rank = 0

        if args.master_addr is None:
            args.master_addr = '127.0.0.1'

        if args.master_port is None:
            args.master_port = get_free_tcp_port()

    return args


@contextlib.contextmanager
def _patch_env(**environs: str):
    """Returns a context manager that patches ``os.environ`` with ``environs``.

    The original ``os.environ`` values are restored at the end.
    """
    # Adapted loosely from https://stackoverflow.com/a/34333710
    # Capture the original environ values
    original_environs = {k: os.environ.get(k) for k in environs}

    # Patch the environment
    for k, v in environs.items():
        os.environ[k] = v
    try:
        # Run the context manager
        yield
    finally:
        # Restore the original environ values
        for k, v in original_environs.items():
            if v is None:
                del os.environ[k]
            else:
                os.environ[k] = v


def _launch_processes(
    nproc: int,
    world_size: int,
    base_rank: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    module_mode: bool,
    command_mode: bool,
    training_script: str,
    stdout_file_format: str,
    stderr_file_format: Union[str, None],
    training_script_args: List[Any],
    processes: Dict[int, subprocess.Popen],
):
    log.info('Starting distributed environment on local node for global_rank(%s-%s)', base_rank, base_rank + nproc - 1)
    log.info('Distributed KV store: tcp://%s:%s', master_addr, master_port)

    for local_rank in range(nproc):
        global_rank = base_rank + local_rank
        if command_mode and module_mode:
            raise ValueError('Either `command_mode` or `module_mode` should be set, but not both.')
        cmd = []
        if not command_mode:
            cmd.append(sys.executable)
        if module_mode:
            cmd.append('-m')

        cmd.append(training_script)

        # Update the env with the distributed variables
        with _patch_env(
            RANK=str(global_rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(local_rank),
            LOCAL_WORLD_SIZE=str(nproc),
            NODE_RANK=str(node_rank),
            MASTER_ADDR=master_addr,
            MASTER_PORT=str(master_port),
            PYTHONUNBUFFERED='1',
            NCCL_ASYNC_ERROR_HANDLING='1',
        ):
            # Populate the distributed variables in all launcher args
            for arg in training_script_args:
                cmd.append(os.path.expandvars(os.path.expanduser(arg)))

            log.info(
                'Launching process for local_rank(%s), global_rank(%s) with command(%s)',
                local_rank,
                global_rank,
                cmd,
            )

            if local_rank == 0:
                process = subprocess.Popen(
                    cmd,
                    text=True,
                )
            else:

                def _get_file(format: str):
                    filename = format.format(
                        rank=global_rank,
                        world_size=world_size,
                        local_rank=local_rank,
                        local_world_size=nproc,
                        node_rank=node_rank,
                    )
                    return open(filename, 'x+')

                stdout_file = _get_file(stdout_file_format)
                stderr_file = _get_file(stderr_file_format) if stderr_file_format is not None else None

                process = subprocess.Popen(
                    cmd,
                    stdout=stdout_file,
                    stderr=stderr_file if stderr_file is not None else subprocess.STDOUT,
                    text=True,
                )
                process.stdout = stdout_file
                if stderr_file is not None:
                    process.stderr = stderr_file
            processes[global_rank] = process


def _monitor_processes(processes: Dict[int, subprocess.Popen]):
    try:
        while True:
            process_has_crashed = False
            all_processes_finished = True
            for global_rank, process in processes.items():
                if process.poll() is None:
                    # the process is still running
                    all_processes_finished = False
                    continue
                else:
                    # return code of 0 implies clean exit
                    if process.returncode != 0:
                        log.error(f'Rank {global_rank} crashed with exit code {process.returncode}.')
                        process_has_crashed = True
                        break
                    else:
                        # exited cleanly
                        log.info(f'Rank {global_rank} finished successfully.')
            if process_has_crashed or all_processes_finished:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('Ctrl-C received; terminating training processes.')
        pass


def _print_process_exit_status(global_rank: int, process: subprocess.Popen):
    stdOutLabel = 'STDOUT'
    if process.stdout is None:
        output = None
    else:
        process.stdout.seek(0)
        output = process.stdout.read()

    if process.stderr is None:
        stderr = None
        stdOutLabel = 'logs'
    else:
        process.stderr.seek(0)
        stderr = process.stderr.read()
    exc = subprocess.CalledProcessError(
        process.returncode,
        cmd=process.args,
        output=output,
        stderr=stderr,
    )

    error_msg = [f'Global rank {global_rank} (PID {process.pid}) exited with code {process.returncode}']
    if output is not None:
        error_msg.extend([
            f'----------Begin global rank {global_rank} {stdOutLabel}----------',
            output,
            f'----------End global rank {global_rank} {stdOutLabel}----------',
        ])

    if stderr is not None:
        error_msg.extend([
            f'----------Begin global rank {global_rank} STDERR----------',
            exc.stderr,
            f'----------End global rank {global_rank} STDERR----------',
        ])
    print('\n'.join(error_msg))


def _cleanup_processes(processes: Dict[int, subprocess.Popen]):
    for global_rank, process in processes.items():
        process.poll()
        if process.returncode is None:
            log.info('Killing global rank %s (PID %s) with SIGTERM', global_rank, process.pid)
            # Assuming that child processes correctly handle SIGTERM to cleanup any children
            try:
                os.kill(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    current_time = datetime.datetime.now()

    try:
        print((
            f'Waiting up to {CLEANUP_TIMEOUT.seconds} seconds for all training processes to terminate. '
            'Press Ctrl-C to exit immediately.'
        ))
        while datetime.datetime.now() - current_time < CLEANUP_TIMEOUT:
            for process in processes.values():
                process.poll()
            if all(process.returncode is not None for process in processes.values()):
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    for global_rank, process in processes.items():
        process.poll()
        if process.returncode is None:
            log.warning(
                'Failed to kill global rank %s (PID %s) with SIGTERM; terminating with SIGKILL instead',
                global_rank,
                process.pid,
            )
            try:
                proc = psutil.Process(process.pid)
            except psutil.NoSuchProcess:
                pass
            else:
                # If using SIGKILL, manually kill all child processes, since the main training process
                # likely won't be able to intercept the signal and clean up its children.
                for psutil_proc in [proc, *proc.children(recursive=True)]:
                    try:
                        os.kill(psutil_proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    for global_rank, process in processes.items():
        process.poll()
        if process.returncode is not None and process.returncode != 0:
            if -process.returncode in (signal.SIGKILL, signal.SIGTERM):
                # Negative return codes indicate the process was killed via a signal
                # If the launcher script killed the training process (which would happen via SIGKILL or SIGTERM),
                # then do not print the stack trace.
                continue
            # only print the processes that have actually crashed,
            # not the ones that were killed
            _print_process_exit_status(global_rank, process)


def _aggregate_process_returncode(processes: Dict[int, subprocess.Popen]) -> int:
    for global_rank, process in processes.items():
        process.poll()
        if process.returncode is None:
            log.error('Global rank %s (PID %s) has still not exited; return exit code 1.', global_rank, process.pid)
            return 1
        if process.returncode != 0:
            log.error('Global rank %s (PID %s) exited with code %s', global_rank, process.pid, process.returncode)
            return process.returncode

    return 0


def main():
    """Entrypoint into the Composer CLI."""
    args = _parse_args()

    logging.basicConfig()
    log.setLevel(logging.INFO if args.verbose else logging.WARNING)

    processes = {}

    log_tmpdir = tempfile.TemporaryDirectory()
    if args.stdout is None:
        args.stdout = f'{log_tmpdir.name}/rank{{rank}}.stdout.txt'
    if args.stderr is None:
        args.stderr = f'{log_tmpdir.name}/rank{{rank}}.stderr.txt'

    # If running on the Mosaic platform, log all gpu ranks' stderr and stdout to Mosaic platform
    if os.environ.get(MOSAICML_PLATFORM_ENV_VAR, 'false').lower() == 'true' and str(
        os.environ.get(MOSAICML_LOG_DIR_ENV_VAR, 'false'),
    ).lower() != 'false' and os.environ.get(MOSAICML_GPU_LOG_FILE_PREFIX_ENV_VAR, 'false').lower() != 'false':
        log.info('Logging all GPU ranks to Mosaic Platform.')
        log_file_format = f'{os.environ.get(MOSAICML_LOG_DIR_ENV_VAR)}/{os.environ.get(MOSAICML_GPU_LOG_FILE_PREFIX_ENV_VAR)}{{local_rank}}.txt'
        if args.stderr is not None or args.stdout is not None:
            log.info(
                'Logging to Mosaic Platform. Ignoring provided stdout and stderr args. To use provided stdout and stderr, set MOSAICML_LOG_DIR=false.',
            )
        args.stdout = log_file_format
        args.stderr = None

    try:
        _launch_processes(
            nproc=args.nproc,
            world_size=args.world_size,
            base_rank=args.base_rank,
            node_rank=args.node_rank,
            master_addr=args.master_addr,
            master_port=args.master_port,
            module_mode=args.module_mode,
            command_mode=args.command_mode,
            stdout_file_format=args.stdout,
            stderr_file_format=args.stderr,
            training_script=args.training_script,
            training_script_args=args.training_script_args,
            processes=processes,
        )
        _monitor_processes(processes)
    except:
        # Print the exception first, then kill the training processes, since killing
        # may take up to CLEANUP_TIMEOUT seconds, and the user should know immediately
        # what failed. No need to re-raise the exception, as `aggregate_process_returncode`
        # will return an appropriate error code, which will cause the script to exit.
        traceback.print_exc()
        print('Killing training processes')
    finally:
        _cleanup_processes(processes)
        log_tmpdir.cleanup()
        return _aggregate_process_returncode(processes)


if __name__ == '__main__':
    sys.exit(main())
