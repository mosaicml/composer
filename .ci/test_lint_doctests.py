# Pytest stub for running lint tests and doctests

# Running these checks through pytest allows us to report any errors in Junit format,
# which is posted directly on the PR

import subprocess
import os
import pytest
import shutil
import pathlib

@pytest.mark.timeout(0)
def test_run_make_lint():
    subprocess.run(["make", "lint"], cwd=os.path.join(os.path.dirname(__file__), ".."), check=True)

@pytest.mark.timeout(0)
def test_run_doctests():
    docs_folder = pathlib.Path(os.path.dirname(__file__)) / '..' / 'docs'
    api_reference_folder = docs_folder / 'source' / 'api_reference'
    # Remove the `api_reference` folder, which isn't automatically removed via `make clean`
    shutil.rmtree(api_reference_folder, ignore_errors=True)
    subprocess.run(["make", "clean"], cwd=docs_folder, check=True)
    # Must build the html first to ensure that doctests in .. autosummary:: generated pages are included
    subprocess.run(["make", "html"], cwd=docs_folder, check=True)
    subprocess.run(["make", "doctest"], cwd=docs_folder, check=True)
