# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

######################
# Base Image Arguments
######################

# CUDA Version
# For a slim CPU-only image, leave the CUDA_VERSION argument blank -- e.g.
# ARG CUDA_VERSION=
ARG CUDA_VERSION=11.3.1

# Calculate the base image based on CUDA_VERSION
ARG BASE_IMAGE=${CUDA_VERSION:+"nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu20.04"}
ARG BASE_IMAGE=${BASE_IMAGE:-"ubuntu:20.04"}

# The Python version to install
ARG PYTHON_VERSION=3.9

# The Pytorch Version to install
ARG PYTORCH_VERSION=1.11.0

# The Torchvision version to install.
# Reference https://github.com/pytorch/vision#installation to determine the Torchvision
# version that corresponds to the PyTorch version
ARG TORCHVISION_VERSION=0.12.0

# In the Dockerimage, Pillow-SIMD is installed instead of Pillow. To trick pip into thinking that
# Pillow is also installed (so it won't override it with a future pip install), a Pillow stub is included
# PILLOW_PSEUDOVERSION is the Pillow version that pip thinks is installed
# PILLOW_SIMD_VERSION is the actual version of pillow-simd that is installed.
ARG PILLOW_PSEUDOVERSION=7.0.0
ARG PILLOW_SIMD_VERSION=7.0.0.post3

# Version of the Mellanox Drivers to install (for InfiniBand support)
# Levave blank for no Mellanox Drivers
ARG MOFED_VERSION=5.5-1.0.3.2

########################
# Vision Image Arguments
########################

# Build the vision image on the pytorch stage
ARG VISION_BASE=pytorch_stage

# Pip version strings of dependencies to install
ARG MMCV_VERSION='==1.4.8'
ARG FFCV_VERSION='==0.0.3'
ARG OPENCV_VERSION='>=4.5.5.64,<4.6'
ARG NUMBA_VERSION='>=0.55.0,<0.56'
ARG MMSEGMENTATION_VERSION='>=0.22.0,<0.23'
ARG CUPY_VERSION='>=10.2.0'

##########################
# Composer Image Arguments
##########################

# Build the composer image on the vision image
ARG COMPOSER_BASE=vision_stage

# The command that is passed to `pip install` -- e.g. `pip install "${COMPOSER_INSTALL_COMMAND}"`
ARG COMPOSER_INSTALL_COMMAND='mosaicml[all]'

#########################
# Build the PyTorch Image
#########################

FROM ${BASE_IMAGE} AS pytorch_stage
ARG DEBIAN_FRONTEND=noninteractive

#######################
# Set the shell to bash
#######################
SHELL ["/bin/bash", "-c"]

ARG CUDA_VERSION

# Remove a bad symlink from the base composer image
# If this file is present after the first command, kaniko
# won't be able to build the docker image.
RUN if [ -n "$CUDA_VERSION" ]; then \
        rm -f /usr/local/cuda-$(echo $CUDA_VERSION | cut -c -4)/cuda-$(echo $CUDA_VERSION | cut -c -4); \
    fi


# update repository keys
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN if [ -n "$CUDA_VERSION" ] ; then \
        rm -f /etc/apt/sources.list.d/cuda.list && \
        rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
        apt-get update &&  \
        apt-get install -y --no-install-recommends wget && \
        apt-get autoclean && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* \
        apt-key del 7fa2af80 && \
        mkdir -p /tmp/cuda-keyring && \
        wget -P /tmp/cuda-keyring https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
        dpkg -i /tmp/cuda-keyring/cuda-keyring_1.0-1_all.deb && \
        rm -rf /tmp/cuda-keyring ; \
    fi

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        wget \
        sudo \
        build-essential \
        git \
        software-properties-common \
        dirmngr \
        apt-utils \
        gpg-agent \
        openssh-client \
        # For PILLOW:
        zlib1g-dev \
        libtiff-dev \
        libfreetype6-dev \
        liblcms2-dev \
        tcl \
        libjpeg8-dev && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

##################################################
# Change the NCCL version to fix NVLink errors
# This is required to train on Nvidia A100s in GCP
##################################################
RUN if [ -n "$CUDA_VERSION" ] ; then \
        apt-get update && \
        apt-get install -y --no-install-recommends --allow-change-held-packages --allow-downgrades \
            libnccl2=2.9.6-1+cuda11.0 \
            libnccl-dev=2.9.6-1+cuda11.0 && \
        apt-get autoclean && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* ; \
    fi

# If using CUDA_VERSION, then use system installed NCCL per update above and point to library
ENV USE_SYSTEM_NCCL=${CUDA_VERSION:+1}
ENV LD_PRELOAD=${CUDA_VERSION:+/usr/lib/x86_64-linux-gnu/libnccl.so.2.9.6}

##############################
# Install NodeJS (for Pyright)
##############################
RUN \
    curl -fsSL https://deb.nodesource.com/setup_17.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

################
# Install Python
################
ARG PYTHON_VERSION

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-apt \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python${PYTHON_VERSION}-venv && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} - && \
    pip${PYTHON_VERSION} install --no-cache-dir --upgrade pip setuptools

#####################
# Install pillow-simd
#####################
ARG PILLOW_PSEUDOVERSION
ARG PILLOW_SIMD_VERSION

# pillow_stub tricks pip into thinking that it installed pillow,
# so when pillow_simd is installed, other packages won't later override it
COPY pillow_stub /tmp/pillow_stub

RUN pip${PYTHON_VERSION} install --no-cache-dir --upgrade /tmp/pillow_stub && \
    pip${PYTHON_VERSION} install --no-cache-dir --upgrade pillow_simd==${PILLOW_SIMD_VERSION} && \
    rm -rf /tmp/pillow_stub

#################
# Install Pytorch
#################
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION

RUN CUDA_VERSION_TAG=$(python${PYTHON_VERSION} -c "print('cu' + ''.join('${CUDA_VERSION}'.split('.')[:2]) if '${CUDA_VERSION}' else 'cpu')") && \
    pip${PYTHON_VERSION} install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
        torch==${PYTORCH_VERSION}+${CUDA_VERSION_TAG} \
        torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION_TAG}

###################################
# Mellanox OFED driver installation
###################################

ARG MOFED_VERSION

RUN if [ -n "$MOFED_VERSION" ] ; then \
        mkdir -p /tmp/mofed && \
        wget -nv -P /tmp/mofed http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu20.04-x86_64.tgz && \
        tar -zxvf /tmp/mofed/MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu20.04-x86_64.tgz -C /tmp/mofed && \
        /tmp/mofed/MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu20.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force && \
        rm -rf /tmp/mofed ; \
    fi


#####################
# Install NVIDIA Apex
#####################
RUN if [ -n "$CUDA_VERSION" ] ; then \
        mkdir -p /tmp/apex && \
        cd /tmp/apex && \
        git clone https://github.com/NVIDIA/apex && \
        cd apex && \
        pip${PYTHON_VERSION} install --no-cache-dir \
            --global-option="--cpp_ext" \
            --global-option="--cuda_ext" \
            --target  /usr/local/lib/python${PYTHON_VERSION}/dist-packages \
            ./ && \
        rm -rf /tmp/apex ; \
    fi


################################
# Use the correct python version
################################

# Set the default python by creating our own folder and hacking the path
# We don't want to use upgrade-alternatives as that will break system packages

ARG COMPOSER_PYTHON_BIN=/composer-python

RUN mkdir -p ${COMPOSER_PYTHON_BIN} && \
    ln -s $(which python${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/python && \
    ln -s $(which python${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/python3 && \
    ln -s $(which python${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/python${PYTHON_VERSION} && \
    ln -s $(which pip${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/pip && \
    ln -s $(which pip${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/pip3 && \
    ln -s $(which pip${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/pip${PYTHON_VERSION} && \
    # Include this folder, and the local bin folder, on the path
    echo "export PATH=~/.local/bin:$COMPOSER_PYTHON_BIN:$PATH" >> /etc/profile && \
    echo "export PATH=~/.local/bin:$COMPOSER_PYTHON_BIN:$PATH" >> /etc/bash.bashrc && \
    echo "export PATH=~/.local/bin:$COMPOSER_PYTHON_BIN:$PATH" >> /etc/zshenv

# Ensure that non-interactive shells load /etc/profile
ENV BASH_ENV=/etc/profile

#########################
# Configure non-root user
#########################
RUN useradd -rm -d /home/mosaicml -s /bin/bash -u 1000 -U -s /bin/bash mosaicml && \
    usermod -a -G sudo mosaicml && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

######################
# PyTorch Vision Image
######################

FROM ${VISION_BASE} AS vision_stage
ARG DEBIAN_FRONTEND=noninteractive

RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    # For FFCV:
    pkg-config \
    libturbojpeg-dev \
    libopencv-dev \
    # For deeplabv3:
    ffmpeg \
    libsm6 \
    libxext6 && \
    sudo apt-get autoclean && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

ARG MMCV_VERSION
ARG FFCV_VERSION
ARG OPENCV_VERSION
ARG NUMBA_VERSION
ARG MMSEGMENTATION_VERSION
ARG PYTHON_VERSION
ARG CUPY_VERSION
ARG CUDA_VERSION

RUN CUDA_VERSION_TAG=$(python${PYTHON_VERSION} -c "print('cu' + ''.join('${CUDA_VERSION}'.split('.')[:2]) if '${CUDA_VERSION}' else 'cpu')") && \
    MMCV_TORCH_VERSION=$(python -c "print('torch' + ''.join('${PYTORCH_VERSION}'.split('.')[:2]) + '.0')") && \
    sudo pip${PYTHON_VERSION} install --no-cache-dir \
        "ffcv${FFCV_VERSION}" \
        "opencv-python${OPENCV_VERSION}" \
        "numba${NUMBA_VERSION}" \
        "mmsegmentation${MMSEGMENTATION_VERSION}" && \
    sudo pip${PYTHON_VERSION} install --no-cache-dir \
        "mmcv-full${MMCV_VERSION}" \
        -f https://download.openmmlab.com/mmcv/dist/${CUDA_VERSION_TAG}/${MMCV_TORCH_VERSION}/index.html && \
    if [ -n "$CUDA_VERSION" ] ; then \
        sudo pip${PYTHON_VERSION} install --no-cache-dir cupy-`echo ${CUDA_VERSION_TAG} | sed "s/cu/cuda/g"`${CUPY_VERSION}; \
    fi


################
# Composer Image
################

FROM ${COMPOSER_BASE} as composer_stage

ARG DEBIAN_FRONTEND=noninteractive

##################
# Install Composer
##################

ARG COMPOSER_INSTALL_COMMAND

RUN pip install "${COMPOSER_INSTALL_COMMAND}"
