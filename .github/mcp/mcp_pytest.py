# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A hello world script using MCP."""

import argparse

from mcli.sdk import RunConfig, create_run, follow_run_logs, wait_for_run_status

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='mcp_pytest', help='Name of run')
parser.add_argument('--platform', type=str, default='r1z1', help='Platform to use')
parser.add_argument('--gpu_type', type=str, default='a100_80gb', help='Type of GPU to use')
parser.add_argument('--gpu_num', type=int, default=8, help='Number of the GPU to use')
parser.add_argument('--image', type=str, default='mosaicml/pytorch:latest', help='Docker image to use')
parser.add_argument('--git_branch', type=str, default='master', help='Name of the git branch to use')
parser.add_argument('--pytest_markers', type=str, help='Markers to pass to pytest')
parser.add_argument('--pytest_command', type=str, help='Command to run pytest')
args = parser.parse_args()

command = f'''
export COMMON_ARGS="-v --durations=20 -m '{args.pytest_markers}'"

git config --global --add safe.directory /__w/composer/composer

ls

cat Makefile

make test PYTEST='{args.pytest_command}' EXTRA_ARGS="$COMMON_ARGS --codeblocks"

make test-dist PYTEST='{args.pytest_command}' EXTRA_ARGS="$COMMON_ARGS" WORLD_SIZE=2

python -m coverage combine

python -m coverage report
'''

config = RunConfig(
    name=args.name,
    platform=args.platform,
    gpu_type=args.gpu_type,
    gpu_num=args.gpu_num,
    image=args.image,
    integrations=[{
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/composer',
        'git_branch': args.git_branch,
        'ssh_clone': 'False',
        'pip_install': '--user -e .[all]',
    }],
    command=command,
)

# Create run
run = create_run(config)

# Wait till run starts before fetching logs
run = wait_for_run_status(run, status='running')

# Print logs
for line in follow_run_logs(run):
    print(line.rstrip('\n'))

# delete_runs([run])
