#!/bin/bash
set -euo pipefail

# This script builds composer as a conda package
# As part of the build process, the composer tests
# are executed. See `meta.yaml` for the conda package
# configuration

# Only use 'sudo' to run apt commands if not already root
SUDO="sudo"
if [ "$UID" == "0" ]; then
    SUDO=""
fi

CONDA_PATH=$HOME/miniconda

# Download and install miniconda if it's not installed
echo "Checking to see if conda is installed"
if ! command -v conda &> /dev/null ; then
    echo "Downloading and installing curl"
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends curl ca-certificates
    echo "Downloading and installing conda"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash /dev/stdin -bfp $CONDA_PATH
else
    echo "Conda is already installed"
    CONDA_PATH=$(conda info --base)
fi

echo "Sourcing conda"
source $CONDA_PATH/etc/profile.d/conda.sh

echo "Checking to see if the 'composer' conda environment exists"
if [[ "$(conda info --envs)" != *"composer"* ]];then
    # If we don't already have a composer conda environment, create it
    echo "Creating the 'composer' conda environment"
    conda create -y composer
else
    echo "The 'composer' conda environment already exists"
fi

# Activate this environment
echo "Activating the composer conda environment"
conda activate composer

# Prepare the conda package
echo "Adding 'mosaicml' to the conda channels"
conda config --add channels mosaicml
echo "Adding 'pytorch' to the conda channels"
conda config --append channels pytorch
echo "Adding 'acaconda' to the conda channels"
conda config --append channels anaconda
echo "Adding 'conda-forge' to the conda channels"
conda config --append channels conda-forge

# Install dependencies
echo "Installing build dependencies"
conda install -y conda-build conda-verify anaconda-client

# Build (without uploading) composer
# Conda-build invokes pytest automatically, and runs all non-gpu tests
echo "Building composer"

conda-build $(dirname $0)/..

# --user mosaicml --token MOSAICML_API_TOKEN $(dirname $0)/..
