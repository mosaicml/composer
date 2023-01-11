# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A hello world script using MCP."""

from mcli.sdk import RunConfig, create_run, follow_run_logs, wait_for_run_status

config = RunConfig(
    name='hello-world',
    platform='r1z1',
    gpu_type='a100_80gb',
    gpu_num=8,
    image='mosaicml/pytorch',
    integrations=[{
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/composer',
        'git_branch': 'dev',
        'ssh_clone': 'False',
        'pip_install': '--user -e .[all]',
    }],
    command='composer_collect_env',
)

run = create_run(config)
print(f'Launching run {run.name}')

run = wait_for_run_status(run, status='running')

# Print logs
for line in follow_run_logs(run):
    print(line.rstrip('\n'))

# delete_runs([run])
