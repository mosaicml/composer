# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Pytest stub for running lint tests and doctests

# Running these checks through pytest allows us to report any errors in Junit format,
# which is posted directly on the PR

import json
import os
import pathlib
import shutil
import subprocess
import textwrap

import pytest


def check_output(proc: subprocess.CompletedProcess):
    # Check the subprocess output, and raise an exception with the stdout/stderr dump if there was a non-zero exit
    # The `check=True` flag available in `subprocess.run` does not print stdout/stderr
    if proc.returncode == 0:
        return
    error_msg = textwrap.dedent(f"""\
        Command {proc.args} failed with exit code {proc.returncode}.
        ----Begin stdout----
        {proc.stdout}
        ----End stdout------
        ----Begin stderr----
        {proc.stderr}
        ----End stderr------""")

    raise RuntimeError(error_msg)


@pytest.mark.timeout(0)
def test_run_pre_commit_hooks():
    composer_root = os.path.join(os.path.dirname(__file__), '..')
    check_output(
        subprocess.run(
            ['pre-commit', 'run', '--all-files', '--show-diff-on-failure'],
            cwd=composer_root,
            capture_output=True,
            text=True,
        ))


@pytest.mark.timeout(0)
def test_run_doctests():
    docs_folder = pathlib.Path(os.path.dirname(__file__)) / '..' / 'docs'
    api_reference_folder = docs_folder / 'source' / 'api_reference'
    # Remove the `api_reference` folder, which isn't automatically removed via `make clean`
    shutil.rmtree(api_reference_folder, ignore_errors=True)
    check_output(subprocess.run(['make', 'clean'], cwd=docs_folder, capture_output=True, text=True))
    # Must build the html first to ensure that doctests in .. autosummary:: generated pages are included
    check_output(subprocess.run(['make', 'html'], cwd=docs_folder, capture_output=True, text=True))
    check_output(subprocess.run(['make', 'doctest'], cwd=docs_folder, capture_output=True, text=True))


@pytest.mark.timeout(30)
def test_docker_build_matrix():
    """Test that the docker build matrix is up to date."""
    docker_folder = pathlib.Path(os.path.dirname(__file__)) / '..' / 'docker'

    # Capture the existing readme and build matrix contents
    with open(docker_folder / 'README.md', 'r') as f:
        existing_readme = f.read()

    with open(docker_folder / 'build_matrix.yaml', 'r') as f:
        existing_build_matrix = f.read()

    # Run the script
    check_output(
        subprocess.run(['python', 'generate_build_matrix.py'], cwd=docker_folder, capture_output=True, text=True))

    # Assert that the files did not change

    with open(docker_folder / 'README.md', 'r') as f:
        assert existing_readme == f.read()

    with open(docker_folder / 'build_matrix.yaml', 'r') as f:
        assert existing_build_matrix == f.read()


@pytest.mark.timeout(30)
def test_json_schemas():
    """Test JSON Schemas are up to date."""
    schemas_folder = pathlib.Path(os.path.dirname(__file__)) / '..' / 'composer' / 'json_schemas'

    # Capture existing schemas
    existing_schemas = []
    for filename in sorted(os.listdir(schemas_folder)):
        if filename != 'generate_json_schemas.py':
            with open(os.path.join(schemas_folder, filename), 'r') as f:
                existing_schemas.append(json.load(f))

    # Run the script
    check_output(
        subprocess.run(['python', 'generate_json_schemas.py'], cwd=schemas_folder, capture_output=True, text=True))

    def sort_json(json):
        if isinstance(json, dict):
            return sorted((key, sort_json(value)) for key, value in json.items())
        elif isinstance(json, list):
            return sorted(sort_json(elem) for elem in json)
        return json

    # Assert that the files did not change
    for existing_schema, filename in zip(existing_schemas, sorted(os.listdir(schemas_folder))):
        if filename != 'generate_json_schemas.py':
            with open(os.path.join(schemas_folder, filename), 'r') as f:
                assert sort_json(existing_schema) == sort_json(json.load(f))


@pytest.mark.parametrize('example', [1, 2])
def test_release_tests_reflect_readme(example: int):
    """Test that example_1.py and example_2.py in release_tests reflect the README.md."""
    with open(pathlib.Path(os.path.dirname(__file__)) / '..' / 'README.md', 'r') as f:
        readme_lines = f.readlines()
    example_code_lines = []
    found_begin = False
    started = False
    for i, line in enumerate(readme_lines):
        if f'begin_example_{example}' in line:
            found_begin = True
            continue
        # Wait until we get the ```python for start of code snippet
        if found_begin and not started:
            if line == '```python\n':
                started = True
        # Reached end of code snippet
        elif started and line == '```\n':
            # Code snippet continues
            if i + 2 < len(readme_lines) and '-->\n' == readme_lines[
                    i + 1] and '<!--pytest-codeblocks:cont-->\n' == readme_lines[i + 2]:
                started = False
            # Code snippet ends
            else:
                break
        # Add line
        elif started:
            example_code_lines.append(line)

    example_file = pathlib.Path(os.path.dirname(__file__)) / 'release_tests' / f'example_{example}.py'
    with open(example_file, 'r') as f:
        assert f.readlines() == example_code_lines
