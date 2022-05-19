#!/usr/bin/env bash

# Script to validate that composer was packaged correctly and that simple examples work
# This script uses a base install of composer, where pytest is not available

set -euo pipefail

CWD=$(pwd)

RELEASE_TEST_FOLDER=$(readlink -f $(dirname $0)/release_tests)

TMPDIR=$(mktemp -d -t ci-XXXXXXXXXX)

cd $TMPDIR

# Do some examples from the readme
python $RELEASE_TEST_FOLDER/example_1.py
python $RELEASE_TEST_FOLDER/example_2.py
composer -n 1 $RELEASE_TEST_FOLDER/print_world_size.py

cd $CWD
rm -rf $TMPDIR
