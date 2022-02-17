#!/usr/bin/env bash
set -euxo pipefail

# This script runs the linting checks and executes doctests
# Doctests are executed on the linting worker, so they are
# executed only once on an install with all dependencies

pip install .[all]
make lint

# Must build the html first to ensure that doctests in .. autosummary:: generated pages are included
cd docs && make html && make doctest && cd ..
