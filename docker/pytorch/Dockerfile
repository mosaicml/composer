 ARG BASE_IMAGE

FROM ${BASE_IMAGE} AS pytorch_stage
ARG DEBIAN_FRONTEND=noninteractive

# Remove a bad symlink from the base composer image
# If this file is present after the first command, kaniko
# won't be able to build the docker image.
RUN rm -f /usr/local/cuda-11.3/cuda-11.3

# update repository keys
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG LINUX_DISTRO
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
        wget -P /tmp/cuda-keyring https://developer.download.nvidia.com/compute/cuda/repos/${LINUX_DISTRO}/x86_64/cuda-keyring_1.0-1_all.deb && \
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
ARG PILLOW_PSEUDOVERSION=7.0.0
ARG PILLOW_SIMD_VERSION=7.0.0.post3

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
ARG CUDA_VERSION_TAG

RUN pip${PYTHON_VERSION} install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
        torch==${PYTORCH_VERSION}+${CUDA_VERSION_TAG} \
        torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION_TAG}

###################################
# Mellanox OFED driver installation
###################################

ARG MOFED_VERSION
ARG MOFED_OS_VERSION

RUN if [ -n "$MOFED_VERSION" ] ; then \
        mkdir -p /tmp/mofed && \
        wget -nv -P /tmp/mofed http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS_VERSION}.tgz && \
        tar -zxvf /tmp/mofed/MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS_VERSION}.tgz -C /tmp/mofed && \
        /tmp/mofed/MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS_VERSION}/mlnxofedinstall --user-space-only --without-fw-update --force && \
        rm -rf /tmp/mofed ; \
    fi

#######################
# Set the shell to bash
#######################
SHELL ["/bin/bash", "-c"]


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

###################
# Add a /tmp folder
###################
RUN mkdir -p -m777 /tmp

FROM pytorch_stage AS vision_stage
ARG DEBIAN_FRONTEND=noninteractive

##############################
# MosaicML Vision Dependencies
##############################

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

ARG CUDA_VERSION_TAG
ARG MMCV_VERSION
ARG MMCV_TORCH_VERSION
ARG PYTHON_VERSION

RUN sudo pip${PYTHON_VERSION} install --no-cache-dir \
        ffcv==0.0.3 \
        'opencv-python>=4.5.5.64,<4.6' \
        'numba>=0.55.0,<0.56' \
        'mmsegmentation>=0.22.0,<0.23' && \
    sudo pip${PYTHON_VERSION} install --no-cache-dir \
        mmcv-full==${MMCV_VERSION} \
        -f https://download.openmmlab.com/mmcv/dist/${CUDA_VERSION_TAG}/${MMCV_TORCH_VERSION}/index.html ; \
    if [ -n "$CUDA_VERSION" ] ; then \
        sudo pip${PYTHON_VERSION} install --no-cache-dir cupy-`echo ${CUDA_VERSION_TAG} | sed "s/cu/cuda/g"`>=10.2.0; \
    fi
