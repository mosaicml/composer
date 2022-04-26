# Pytest stub for running lint tests and doctests

# Running these checks through pytest allows us to report any errors in Junit format,
# which is posted directly on the PR

import subprocess
import os
import pytest
import shutil
import pathlib
import textwrap

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
def test_run_make_lint():
    composer_root = os.path.join(os.path.dirname(__file__), "..")
    check_output(subprocess.run(["make", "lint"], cwd=composer_root, capture_output=True, text=True))

@pytest.mark.timeout(0)
def test_run_doctests():
    docs_folder = pathlib.Path(os.path.dirname(__file__)) / '..' / 'docs'
    api_reference_folder = docs_folder / 'source' / 'api_reference'
    # Remove the `api_reference` folder, which isn't automatically removed via `make clean`
    shutil.rmtree(api_reference_folder, ignore_errors=True)
    check_output(subprocess.run(["make", "clean"], cwd=docs_folder, capture_output=True, text=True))
    # Must build the html first to ensure that doctests in .. autosummary:: generated pages are included
    check_output(subprocess.run(["make", "html"], cwd=docs_folder, capture_output=True, text=True))
    check_output(subprocess.run(["make", "doctest"], cwd=docs_folder, capture_output=True, text=True))
