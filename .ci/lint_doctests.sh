#!/usr/bin/env bash
set -euxo pipefail

# This script runs the linting checks and executes doctests
# Doctests are executed on the linting worker, so they are
# executed only once on an install with all dependencies

# Install dependencies
sudo npm install -g pyright@1.1.224
pip install .[all]

# Clean and make the output directory
BUILD_DIR="build/output"
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

# Run lint and doctests through pytest
pytest $(dirname $0)/test_lint_doctests.py --junitxml ${BUILD_DIR}/build${BUILD_NUMBER}_lint_doctests.junit.xml
