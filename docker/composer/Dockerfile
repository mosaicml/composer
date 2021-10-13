FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
ARG COMPOSER_LIB_DEP=base


################################################
# Python v1.8.1
################################################

# Install basic tools, libraries and Python 1.8.1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openssh-client \
    git \
    vim \
    emacs \
    libgomp1 \
    wget \
    gcc \
    libjpeg8-dev \
    htop \
    python3-apt \
    python3.8 \
    python3.8-dev \
    python3.8-distutils && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    wget https://bootstrap.pypa.io/get-pip.py  && \
    python3.8 get-pip.py && \
    rm get-pip.py && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
## only links python as python3.8 as python3.6 is used for some system installs

# Needed to add Nvidia repo and key 
RUN apt-get update && \
    apt-get --no-install-recommends -y install software-properties-common

# Upgrade NCCL to fix NVLink errors, mainly for GCP distributed runs
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt update && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" && \
    apt install -y --allow-change-held-packages libnccl2=2.9.6-1+cuda11.0 libnccl-dev=2.9.6-1+cuda11.0 && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Use system installed NCCL per update above, point to library
ENV USE_SYSTEM_NCCL=1
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.9.6


################################################
# PyTorch v1.9.0
################################################

# Install PyTorch, Torchvision and Cuda toolkit
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111


################################################
# MosaicML Composer
################################################

# Add Github to known hosts
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /workspace

COPY . ./src/composer

# Clear repo information, users can re-create if needed
RUN rm -rf /workspace/src/composer/.git

# Install composer package
RUN pip install -e src/composer[$COMPOSER_LIB_DEP]

# Install accelerated Pillow-SIMD package, default pillow must first be uninstalled per instructions:
# https://github.com/uploadcare/pillow-simd
RUN pip uninstall -y pillow && \
    pip install -U pillow_simd==6.2.2.post1 --no-cache-dir
