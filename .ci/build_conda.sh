#!/usr/bin/env bash

set -euo pipefail

# This script builds composer as a conda package
# As part of the build process, the composer tests
# are executed. See `meta.yaml` for the conda package
# configuration

# Install git and make, which are required to clone the repo and run tests
yum install -y git make

# Prepare the conda package

echo "Adding 'conda-forge' to the conda channels"
conda config --add channels conda-forge
echo "Adding 'pytorch' to the conda channels"
conda config --add channels pytorch
echo "Adding 'mosaicml' to the conda channels"
conda config --add channels mosaicml
echo "Adding 'defaults' to the conda channels"
conda config --add channels defaults

# echo "Setting strict conda channel priority"
# conda config --set channel_priority strict

# Install dependencies
echo "Installing build dependencies"
conda install -y conda-build conda-verify anaconda-client

# Build (without uploading) composer
# Conda-build invokes pytest automatically, and runs all non-gpu tests
echo "Building composer"

conda-build $(dirname $0)/..
