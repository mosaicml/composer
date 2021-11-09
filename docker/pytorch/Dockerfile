ARG BASE_IMAGE

FROM ${BASE_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        build-essential \
        software-properties-common \
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

# # Upgrade NCCL to fix NVLink errors
# # No Need to install, as the cuda 11.3 image already contains a late enough version
# RUN DISTRO="$(lsb_release -si | tr '[:upper:]' '[:lower:]')$(lsb_release -sr | sed 's/\.//')" &&  \
#     apt-get update && \
#     apt-get install -y --no-install-recommends \
#         dirmngr \
#         gpg-agent && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/7fa2af80.pub && \
#     add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/ /" && \
#     apt-get install -y --no-install-recommends --allow-change-held-packages \
#         libnccl2 \
#         libnccl-dev && \
#     apt-get autoclean && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# # Use system installed NCCL per update above, point to library
# ENV USE_SYSTEM_NCCL=1
# ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.9.6

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
    python${PYTHON_VERSION}-distutils && \
    python${PYTHON_VERSION}-venv && \
    python${PYTHON_VERSION}-wheel && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py | python - && \
    pip install --no-cache-dir --upgrade pip

#####################
# Install pillow-simd
#####################
ARG PILLOW_PSEUDOVERSION=7.0.0
ARG PILLOW_SIMD_VERSION=7.0.0.post3

# pillow_stub tricks pip into thinking that it installed pillow,
# so when pillow_simd is installed, other packages won't later override it
COPY pillow_stub /tmp/pillow_stub

RUN pip install --no-cache-dir --upgrade /tmp/pillow_stub && \
    pip install --no-cache-dir --upgrade pillow_simd==${PILLOW_SIMD_VERSION}

#################
# Install Pytorch
#################
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG CUDA_VERSION_TAG

RUN pip install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
        torch==${PYTORCH_VERSION}+${CUDA_VERSION_TAG} \
        torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION_TAG}

#########
# Cleanup
#########
RUN rm -rf /tmp/*

#########################
# Configure non-root user
#########################
RUN echo "export PATH=\$PATH:~/.local/bin" >> /etc/skel/.bashrc && \
    useradd -rm -d /home/mosaicml -s /bin/bash -u 1000 -U -s /bin/bash mosaicml
WORKDIR /home/mosaicml

USER mosaicml
