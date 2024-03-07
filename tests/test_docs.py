# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Pytest stub for running lint tests and doctests

# Running these checks through pytest allows us to report any errors in Junit format,
# which is posted directly on the PR

import os
import subprocess
import textwrap

import pytest


def check_output(proc: subprocess.CompletedProcess):
    # Check the subprocess output, and raise an exception with the stdout/stderr dump if there was a non-zero exit
    # The `check=True` flag available in `subprocess.run` does not print stdout/stderr
    if proc.returncode == 0:
        return
    error_msg = textwrap.dedent(
        f"""\
        Command {proc.args} failed with exit code {proc.returncode}.
        ----Begin stdout----
        {proc.stdout}
        ----End stdout------
        ----Begin stderr----
        {proc.stderr}
        ----End stderr------""",
    )

    raise RuntimeError(error_msg)


@pytest.mark.doctest
def test_run_doctests():
    docs_folder = os.path.join(os.path.dirname(__file__), '..', 'docs')
    check_output(subprocess.run(['make', 'clean'], cwd=docs_folder, capture_output=True, text=True))
    # Must build the html first to ensure that doctests in .. autosummary:: generated pages are included
    check_output(subprocess.run(['make', 'html'], cwd=docs_folder, capture_output=True, text=True))
    check_output(subprocess.run(['make', 'doctest'], cwd=docs_folder, capture_output=True, text=True))


@pytest.mark.doctest
def test_docker_build_matrix():
    """Test that the docker build matrix is up to date."""
    docker_folder = os.path.join(os.path.dirname(__file__), '..', 'docker')

    # Capture the existing readme and build matrix contents
    with open(os.path.join(docker_folder, 'README.md'), 'r') as f:
        existing_readme = f.read()

    with open(os.path.join(docker_folder, 'build_matrix.yaml'), 'r') as f:
        existing_build_matrix = f.read()

    # Run the script
    check_output(
        subprocess.run(
            ['python', os.path.join(docker_folder, 'generate_build_matrix.py')],
            cwd=docker_folder,
            capture_output=True,
            text=True,
        ),
    )

    # Assert that the files did not change

    with open(os.path.join(docker_folder, 'README.md'), 'r') as f:
        assert existing_readme == f.read()

    with open(os.path.join(docker_folder, 'build_matrix.yaml'), 'r') as f:
        assert existing_build_matrix == f.read()
