#!/usr/bin/env bash
set -exuo pipefail

# This script runs pytest and produces coverage and junitxml reports for Jenkins
# It expects the first CLI argument to be the extra dependencies to install
# And the second CLI argument should be the pytest markers

EXTRA_DEPS="$1"
MARKERS="$2"

# Install dependencies
if [ -z "${EXTRA_DEPS}" ]; then
    pip install .
else
    pip install .[${EXTRA_DEPS}]
fi

# Run pytest
JUNIT_PREFIX=build/output/${BUILD_NUMBER}
mkdir -p $(dirname $JUNIT_PREFIX)
python -m coverage run -m pytest --junitxml $JUNIT_PREFIX.n0.junit.xml --duration all -v -m "$MARKERS"
python -m composer.cli.launcher -n 2 --master_port 26000 -m coverage run -m pytest --junitxml $JUNIT_PREFIX.n2.junit.xml --duration all -v -m "$MARKERS"

# Combine the coverage reports
python -m coverage combine
python -m coverage xml -o build/output/${BUILD_NUMBER}.coverage.xml
