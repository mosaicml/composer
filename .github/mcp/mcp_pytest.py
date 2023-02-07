# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Run pytest using MCP."""

import argparse
from concurrent.futures import TimeoutError

from mcli.sdk import RunConfig, RunStatus, create_run, get_run_logs, stop_runs, wait_for_run_status

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='mcp-pytest', help='Base name of run')
    parser.add_argument('--cluster', type=str, default='r1z4', help='Cluster to use')
    parser.add_argument('--gpu_type', type=str, default='a100_40gb', help='Type of GPU to use')
    parser.add_argument('--gpu_num', type=int, default=2, help='Number of the GPU to use')
    parser.add_argument('--image', type=str, default='mosaicml/pytorch:latest', help='Docker image to use')
    parser.add_argument('--git_branch', type=str, help='Git branch to check out')
    parser.add_argument('--git_commit', type=str, help='Git commit to check out. Overrides git_branch if specified')
    parser.add_argument('--pr_number',
                        type=int,
                        help='PR number to check out. Overrides git_branch/git_commit if specified')
    parser.add_argument('--pytest_markers', type=str, help='Markers to pass to pytest')
    parser.add_argument('--pytest_command', type=str, help='Command to run pytest')
    parser.add_argument('--timeout', type=int, default=1800, help='Timeout for run (in seconds)')
    args = parser.parse_args()

    name = args.name
    git_integration = {
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/composer',
        'ssh_clone': 'False',
    }
    if args.git_branch is not None and args.git_commit is None:
        name += f'-branch-{args.git_branch}'
        git_integration['git_branch'] = args.git_branch
    if args.git_commit is not None:
        name += f'-commit-{args.git_commit}'
        git_integration['git_commit'] = args.git_commit

    command = 'cd composer'

    # Checkout a specific PR if specified
    if args.pr_number is not None:
        name += f'-pr-{args.pr_number}'
        command += f'''

        git fetch origin pull/{args.pr_number}/head:pr_branch

        git checkout pr_branch

        '''

    # Shorten name if too long
    if len(name) > 56:
        name = name[:56]

    command += f'''

    pip install --upgrade --user .[all]

    export COMMON_ARGS="-v --durations=20 -m '{args.pytest_markers}'"

    make test PYTEST='{args.pytest_command}' EXTRA_ARGS="$COMMON_ARGS --codeblocks"

    make test-dist PYTEST='{args.pytest_command}' EXTRA_ARGS="$COMMON_ARGS" WORLD_SIZE=2

    python -m coverage combine

    python -m coverage report
    '''

    config = RunConfig(
        name=name,
        cluster=args.cluster,
        gpu_type=args.gpu_type,
        gpu_num=args.gpu_num,
        image=args.image,
        integrations=[git_integration],
        command=command,
    )

    # Create run
    run = create_run(config)
    print(f'Run created: {run.name}')

    # Wait until run starts before fetching logs
    run = wait_for_run_status(run, status='running')
    print('Run started. Waiting for run to complete...')

    # Wait up to args.timeout seconds for run to complete
    try:
        run = wait_for_run_status(run, status='completed', timeout=args.timeout)
    except TimeoutError:
        print(f'Run timed out and did not complete in {args.timeout/60} minutes.')

    # Get run status and stop run
    success = run.status == RunStatus.COMPLETED
    print(f'Run completed with status: {run.status} (success={success})')
    if run.status == RunStatus.RUNNING:
        stop_runs([run])
        print('Run stopped.')

    # Print logs
    for line in get_run_logs(run):
        print(line, end='')

    # Fail if command exited with non-zero exit code or timed out
    assert success
