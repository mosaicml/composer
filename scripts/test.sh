#!/bin/bash
set -exuo pipefail

JUNIT_PREFIX=${JUNIT_PREFIX:-'build/output/composer'}

mkdir -p $(dirname $JUNIT_PREFIX)

python -m composer.cli.launcher -n 1 --master_port 26000 -m coverage run -m pytest --junitxml $JUNIT_PREFIX.n1.junit.xml $@
python -m composer.cli.launcher -n 2 --master_port 26000 -m coverage run -m pytest --junitxml $JUNIT_PREFIX.n2.junit.xml $@
