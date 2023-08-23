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
ARG PYTHON_VERSION=3.10

# The Pytorch Version to install
ARG PYTORCH_VERSION=1.13.1

# The Torchvision version to install.
# Reference https://github.com/pytorch/vision#installation to determine the Torchvision
# version that corresponds to the PyTorch version
ARG TORCHVISION_VERSION=0.14.1

# The Torchtext version to install.
# Reference https://github.com/pytorch/text#installation to determine the Torchtext
# version that corresponds to the PyTorch version
ARG TORCHTEXT_VERSION=0.14.1

# In the Dockerimage, Pillow-SIMD is installed instead of Pillow. To trick pip into thinking that
# Pillow is also installed (so it won't override it with a future pip install), a Pillow stub is included
# PILLOW_PSEUDOVERSION is the Pillow version that pip thinks is installed
# PILLOW_SIMD_VERSION is the actual version of pillow-simd that is installed.
ARG PILLOW_PSEUDOVERSION=9.3.0
ARG PILLOW_SIMD_VERSION=9.0.0.post1

# Version of the Mellanox Drivers to install (for InfiniBand support)
# Leave blank for no Mellanox Drivers
ARG MOFED_VERSION=5.5-1.0.3.2

# Version of EFA Drivers to install (for AWS Elastic Fabric Adapter support)
# Leave blank for no EFA Drivers
ARG AWS_OFI_NCCL_VERSION=v1.5.0-aws

# Upgrade certifi to resolve CVE-2022-23491
ARG CERTIFI_VERSION='>=2022.12.7'

# Upgrade ipython to resolve CVE-2023-24816
ARG IPYTHON_VERSION='>=8.10.0'

# Upgrade urllib to resolve CVE-2021-33503
ARG URLLIB3_VERSION='>=1.26.5,<2'

########################
# Vision Image Arguments
########################

# Build the vision image on the pytorch stage
ARG VISION_BASE=pytorch_stage

# Pip version strings of dependencies to install
ARG MMCV_VERSION='==1.4.8'
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
        libjpeg8-dev \
        less \
        # For AWS EFA:
        autoconf \
        autotools-dev \
        automake \
        libtool \
        # Development tools
        tmux \
        htop && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

###############################
# Install latest version of git
###############################
RUN add-apt-repository ppa:git-core/ppa && \
    apt-get install -y --no-install-recommends \
        git && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

##############################
# Install NodeJS (for Pyright)
##############################
RUN \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

################
# Install Python
################
ARG PYTHON_VERSION

# Python 3.10 changes where packages are installed. Workaround until all packages support new format
ENV DEB_PYTHON_INSTALL_LAYOUT=deb

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
    pip${PYTHON_VERSION} install --no-cache-dir --upgrade 'pip<23' setuptools

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
ARG PYTORCH_NIGHTLY_URL
ARG PYTORCH_NIGHTLY_VERSION
ARG TORCHVISION_VERSION
ARG TORCHTEXT_VERSION

# Set so supporting PyTorch packages such as Torchvision, Torchaudio, Torchtext pin PyTorch version
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
ENV PYTORCH_NIGHTLY_URL=${PYTORCH_NIGHTLY_URL}
ENV PYTORCH_NIGHTLY_VERSION=${PYTORCH_NIGHTLY_VERSION}

RUN if [ -z "$PYTORCH_NIGHTLY_URL" ] ; then \
      CUDA_VERSION_TAG=$(python${PYTHON_VERSION} -c "print('cu' + ''.join('${CUDA_VERSION}'.split('.')[:2]) if '${CUDA_VERSION}' else 'cpu')") && \
        pip${PYTHON_VERSION} install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
            torch==${PYTORCH_VERSION}+${CUDA_VERSION_TAG} \
            torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION_TAG} \
            torchtext==${TORCHTEXT_VERSION} ; \
    else \
        pip${PYTHON_VERSION} install --no-cache-dir --pre --index-url ${PYTORCH_NIGHTLY_URL} \
            torch==${PYTORCH_VERSION}.${PYTORCH_NIGHTLY_VERSION} \
	    torchvision==${TORCHVISION_VERSION}.${PYTORCH_NIGHTLY_VERSION} \
            torchtext ; \
    fi
RUN
#####################################
# Install EFA and AWS-OFI-NCCL plugin
#####################################

ARG EFA_INSTALLER_VERSION=latest
ARG AWS_OFI_NCCL_VERSION

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:$PATH
ENV FI_EFA_USE_DEVICE_RDMA=1

RUN if [ -n "$AWS_OFI_NCCL_VERSION" ] ; then \
        cd /tmp && \
        curl -OsS https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
        tar -xf /tmp/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
        cd aws-efa-installer && \
        apt-get update && \
        ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify && \
        rm -rf /tmp/aws-efa-installer* ; \
    fi

RUN if [ -n "$AWS_OFI_NCCL_VERSION" ] ; then \
        git clone https://github.com/aws/aws-ofi-nccl.git /opt/aws-ofi-nccl && \
        cd /opt/aws-ofi-nccl && \
        git checkout ${AWS_OFI_NCCL_VERSION} && \
        ./autogen.sh && \
        ./configure --prefix=/opt/aws-ofi-nccl/install \
            --with-libfabric=/opt/amazon/efa/ \
            --with-cuda=/usr/local/cuda \
            --disable-tests && \
        make && make install ; \
    fi

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
# skip if torch nightly is installed as there is incompatability
RUN if [[ -n "$CUDA_VERSION" ]] &&  [[ -z "${PYTORCH_NIGHTLY_URL}" ]]; then \
        mkdir -p /tmp/apex && \
        cd /tmp/apex && \
        git clone https://github.com/NVIDIA/apex && \
        cd apex && \
        git checkout 82ee367f3da74b4cd62a1fb47aa9806f0f47b58b && \
        pip${PYTHON_VERSION} install --no-cache-dir -r requirements.txt && \
        pip${PYTHON_VERSION} install --no-cache-dir \
            --global-option="--cpp_ext" \
            --global-option="--cuda_ext" \
            --target  /usr/local/lib/python${PYTHON_VERSION}/dist-packages \
            ./ && \
        rm -rf /tmp/apex ; \
    fi

##########################
# Install Flash Attention
##########################
RUN if [ -n "$CUDA_VERSION" ] ; then \
        pip${PYTHON_VERSION} install --upgrade --no-cache-dir ninja==1.11.1 && \
        pip${PYTHON_VERSION} install --upgrade --no-cache-dir --force-reinstall packaging==22.0 && \
        pip${PYTHON_VERSION} install --no-cache-dir flash-attn==1.0.3.post0; \
    fi

###############
# Install cmake
###############
RUN pip${PYTHON_VERSION} install --no-cache-dir cmake==3.26.3

###########################
# Install Pandoc Dependency
###########################
# Pandoc is needed for the documentation build and is a nuisance to install via pip so just installing via dpkg
RUN wget https://github.com/jgm/pandoc/releases/download/2.19.2/pandoc-2.19.2-1-amd64.deb && \
    dpkg -i pandoc-2.19.2-1-amd64.deb && \
    rm pandoc-2.19.2-1-amd64.deb

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

#########################
# Upgrade apt packages
#########################
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#########################
# Upgrade pip packages
#########################
RUN pip install --no-cache-dir --upgrade \
        certifi${CERTIFI_VERSION} \
        ipython${IPYTHON_VERSION} \
        urllib3${URLLIB3_VERSION}


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
ARG OPENCV_VERSION
ARG NUMBA_VERSION
ARG MMSEGMENTATION_VERSION
ARG PYTHON_VERSION
ARG CUPY_VERSION
ARG CUDA_VERSION

RUN CUDA_VERSION_TAG=$(python${PYTHON_VERSION} -c "print('cu' + ''.join('${CUDA_VERSION}'.split('.')[:2]) if '${CUDA_VERSION}' else 'cpu')") && \
    MMCV_TORCH_VERSION=$(python -c "print('torch' + ''.join('${PYTORCH_VERSION}'.split('.')[:2]) + '.0')") && \
    sudo pip${PYTHON_VERSION} install --no-cache-dir \
        "ffcv<1.0.3" \
        "opencv-python${OPENCV_VERSION}" \
        "numba${NUMBA_VERSION}" \
        "mmsegmentation${MMSEGMENTATION_VERSION}" && \
    sudo pip${PYTHON_VERSION} install --no-cache-dir \
        "mmcv-full${MMCV_VERSION}" \
        -f https://download.openmmlab.com/mmcv/dist/${CUDA_VERSION_TAG}/${MMCV_TORCH_VERSION}/index.html && \
    if [ -n "$CUDA_VERSION" ] ; then \
        sudo pip${PYTHON_VERSION} install --no-cache-dir cupy-cuda11x${CUPY_VERSION}; \
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
