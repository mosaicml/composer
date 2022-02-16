#!/usr/bin/env bash
set -exuo pipefail


# This script wraps calls to pytest with the composer launch script
# so that multi-process tests (e.g. DDP) are correctly run.
# It also records coverage and junitxml.
# All CLI arguments to test.sh are passed through directly to pytest.

JUNIT_PREFIX=${JUNIT_PREFIX:-'build/output/composer'}

mkdir -p $(dirname $JUNIT_PREFIX)

# Run single-rank tests without distributed
python -m coverage run -m pytest --junitxml $JUNIT_PREFIX.n0.junit.xml "$@"

# Run dual-rank tests with distributed
python -m composer.cli.launcher -n 2 --master_port 26000 -m coverage run -m pytest --junitxml $JUNIT_PREFIX.n2.junit.xml "$@"
