#!/bin/bash
set -exuo pipefail

# For every pytorch base, build a corresponding composer base

COMPOSER_EXTRA_DEPS_TAGS="base dev all"
COMPOSER_VERSION=$(python $(dirname $0)/../../setup.py --version | xargs)

while read BUILD_ARGS; do
    echo $BUILD_ARGS
    eval $BUILD_ARGS  # sets TAG
    BASE_IMAGE=$TAG
    for COMPOSER_EXTRA_DEPS in ${COMPOSER_EXTRA_DEPS_TAGS[@]}; do
        TAG="mosaicml/composer:${COMPOSER_VERSION}_${COMPOSER_EXTRA_DEPS}-pytorch${PYTORCH_VERSION}_${CUDA_VERSION}-python${PYTHON_VERSION}-${LINUX_DISTRO}"
        echo "BASE_IMAGE=$BASE_IMAGE COMPOSER_EXTRA_DEPS=$COMPOSER_EXTRA_DEPS TAG=$TAG"
    done

done <<< $($(dirname $0)/../pytorch/build_matrix.sh)
