# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A hello world script using MCP."""

from mcli.sdk import RunConfig, create_run, follow_run_logs

config = RunConfig(
    name='hello-world',
    gpu_type='a100_80gb',
    platform='r1z1',
    image='mosaicml/pytorch',
    command='composer_collect_env',
)

run = create_run(config)
print(f'Launching run {run.name}')

# Print logs
for line in follow_run_logs(run):
    print(f'[Log]: {line}')

# delete_runs([run])
