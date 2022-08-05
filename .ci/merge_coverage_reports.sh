#!/usr/bin/env bash
set -exuo pipefail

BUILD_OUTPUT_FOLDER="$1"
MERGED_COVERAGE_FILE="$2"

set +e

cp ${BUILD_OUTPUT_FOLDER}/.coverage* ./
ls -al .coverage*
EXIT_CODE="$?"
set -e
if [[ "$EXIT_CODE" != 0 ]]; then
    echo "No coverage files found"
else
    pip install 'coverage[toml]==6.4.2'

    python -m coverage combine .coverage*

    python -m coverage xml -o ${MERGED_COVERAGE_FILE}

    mv .coverage ${BUILD_OUTPUT_FOLDER}/.coverage
fi
