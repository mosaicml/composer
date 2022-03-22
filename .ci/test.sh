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

# Clean and make the output directory
BUILD_DIR="build/output"
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}/${BUILD_NUMBER}_n0
mkdir -p ${BUILD_DIR}/${BUILD_NUMBER}_n2

function cleanup()
{
    # Remove all the non-log test artifacts
    # Don't want to upload model checkpoints, etc... to Jenkins, as that will quickly cause the artifact store
    # to run out of space.
    rm -rf ${BUILD_DIR}/${BUILD_NUMBER}_n0/test_*
    rm -rf ${BUILD_DIR}/${BUILD_NUMBER}_n2/test_*

    # Combine the coverage reports
    python -m coverage combine
    python -m coverage xml -o build/output/${BUILD_NUMBER}.coverage.xml
}

trap cleanup EXIT

# Set the run directory to build/output, which will be caputred by Jenkins
# Run pytest with coverage, and store the junit output
make test \
    PYTHON="COMPOSER_RUN_DIRECTORY=${BUILD_DIR}/${BUILD_NUMBER}_n0 python" \
    PYTEST="coverage run -m pytest" \
    DURATION=all \
    EXTRA_ARGS="--junitxml ${BUILD_DIR}/${BUILD_NUMBER}.n0.junit.xml -v -m '$MARKERS'"

RANK_ARG='\$${RANK}' # escape RANK from the makefile and the makefile shell command
make test-dist \
    PYTEST="coverage run -m pytest" \
    DURATION=all \
    WORLD_SIZE=2 \
    EXTRA_LAUNCHER_ARGS="--run_directory ${BUILD_DIR}/${BUILD_NUMBER}_n2" \
    EXTRA_ARGS="--junitxml ${BUILD_DIR}/${BUILD_NUMBER}.${RANK_ARG}_n2.junit.xml -v -m '$MARKERS'"
