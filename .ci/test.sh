#!/usr/bin/env bash
set -exuo pipefail

# This script runs pytest and produces coverage and junitxml reports for Jenkins
# It expects the first CLI argument to be the extra dependencies to install
# And the second CLI argument should be the pytest markers

EXTRA_DEPS="$1"
MARKERS="$2"

# Integration test settings
export WANDB_ENTITY='mosaicml-public-integration-tests'
export WANDB_PROJECT="integration-tests-${BUILD_NUMBER}-$(date +%s)"
S3_BUCKET='mosaicml-internal-integration-testing'
SFTP_URI='sftp://mosaicml-integration-testing@s-c07c6cb0dd1441dbb.server.transfer.us-west-2.amazonaws.com/mosaicml-internal-integration-testing'


# Install dependencies
if [ -z "${EXTRA_DEPS}" ]; then
    pip install .
else
    pip install ".[${EXTRA_DEPS}]"
fi

# Clean and make the output directory
BUILD_DIR="build/output"
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

function cleanup()
{
    # Combine the coverage reports
    python -m coverage combine
    python -m coverage xml -o build/output/build${BUILD_NUMBER}.coverage.xml

    # Move the raw merged .coverage file into the build artifact folder
    mv .coverage ${BUILD_DIR}/.coverage.${BUILD_NUMBER}
}

trap cleanup EXIT

COMMON_ARGS="-v -m '$MARKERS' --s3_bucket '$S3_BUCKET' --sftp_uri '$SFTP_URI'"

# Set the run directory to build/output, which will be caputred by Jenkins
# Run pytest with coverage, and store the junit output
make test \
    PYTEST="coverage run -m pytest" \
    EXTRA_ARGS="--codeblocks --junitxml ${BUILD_DIR}/build${BUILD_NUMBER}_nproc0.junit.xml $COMMON_ARGS"

RANK_ARG='\$${RANK}' # escape RANK from the makefile and the makefile shell command
make test-dist \
    PYTEST="coverage run -m pytest" \
    WORLD_SIZE=2 \
    EXTRA_LAUNCHER_ARGS="--stdout ${BUILD_DIR}/build${BUILD_NUMBER}_nproc2_rank{rank}.stdout.txt \
        --stderr ${BUILD_DIR}/build${BUILD_NUMBER}_nproc2_rank{rank}.stderr.txt" \
    EXTRA_ARGS="--junitxml ${BUILD_DIR}/build${BUILD_NUMBER}_rank${RANK_ARG}_nproc2.junit.xml $COMMON_ARGS"
