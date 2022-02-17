#!/bin/bash
set -euo pipefail

# This script builds composer as a conda package
# As part of the build process, the composer tests
# are executed. See `meta.yaml` for the conda package
# configuration

# Download and install miniconda if it's not installed
echo "Checking to see if conda is installed"
if [ ! command -v conda ] &> /dev/null ; then
    echo "Downloading and installing conda"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash /dev/stdin -bfp $HOME/miniconda
else
    echo "Conda is already installed"
fi

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
source $(conda info --base)/etc/profile.d/conda.sh
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
