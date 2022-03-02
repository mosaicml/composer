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
make test PYTEST="coverage run -m pytest" DURATION=all EXTRA_ARGS="--junitxml $JUNIT_PREFIX.n0.junit.xml -v -m '$MARKERS'"
RANK_ARG='\$${RANK}' # escape RANK from the makefile and the makefile shell command
make test-dist PYTEST="coverage run -m pytest" DURATION=all WORLD_SIZE=2 EXTRA_ARGS="--junitxml $JUNIT_PREFIX.${RANK_ARG}_n2.junit.xml -v -m '$MARKERS'"

# Combine the coverage reports
python -m coverage combine
python -m coverage xml -o build/output/${BUILD_NUMBER}.coverage.xml
