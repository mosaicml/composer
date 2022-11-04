# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Pytest stub for running lint tests and doctests

# Running these checks through pytest allows us to report any errors in Junit format,
# which is posted directly on the PR

import os
import subprocess
import textwrap


def check_output(proc: subprocess.CompletedProcess):
    # Check the subprocess output, and raise an exception with the stdout/stderr dump if there was a non-zero exit
    # The `check=True` flag available in `subprocess.run` does not print stdout/stderr
    error_msg = textwrap.dedent(f"""\
        Command {proc.args} failed with exit code {proc.returncode}.
        ----Begin stdout----
        {proc.stdout}
        ----End stdout------
        ----Begin stderr----
        {proc.stderr}
        ----End stderr------""")
    print(error_msg)
    if proc.returncode == 0:
        return

    raise RuntimeError(error_msg)


def test_run_pre_commit_hooks():
    composer_root = os.path.join(os.path.dirname(__file__), '..')
    check_output(
        subprocess.run(
            ['pre-commit', 'run', '--all-files', '--show-diff-on-failure'],
            cwd=composer_root,
            capture_output=True,
            text=True,
        ))
