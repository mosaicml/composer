FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
ARG DEBIAN_FRONTEND=noninteractive

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
# only links python as python3.8 as python3.6 is used for some system installs

RUN apt-get update && \
    apt-get --no-install-recommends -y install software-properties-common

# Upgrade NCCL to fix NVLink errors during distributed runs
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt update && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" && \
    apt install -y --allow-change-held-packages libnccl2=2.9.6-1+cuda11.0 libnccl-dev=2.9.6-1+cuda11.0 && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV USE_SYSTEM_NCCL=1
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.9.6

