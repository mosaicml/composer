ARG BASE_IMAGE

FROM ${BASE_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive

# remove a bad symlink
RUN rm -f /usr/local/cuda-11.3/cuda-11.3  

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        sudo \
        build-essential \
        git \
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

##################################################
# Change the NCCL version to fix NVLink errors
# This is required to train on Nvidia A100s in GCP
##################################################
RUN DISTRO="$(lsb_release -si | tr '[:upper:]' '[:lower:]')$(lsb_release -sr | sed 's/\.//')" &&  \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        dirmngr \
        gpg-agent && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/7fa2af80.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/ /" && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        libnccl2=2.9.6-1+cuda11.0 \
        libnccl-dev=2.9.6-1+cuda11.0 && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Use system installed NCCL per update above, point to library
ENV USE_SYSTEM_NCCL=1
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.9.6

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
    pip${PYTHON_VERSION} install --no-cache-dir --upgrade pip

#####################
# Install pillow-simd
#####################
ARG PILLOW_PSEUDOVERSION=7.0.0
ARG PILLOW_SIMD_VERSION=7.0.0.post3

# pillow_stub tricks pip into thinking that it installed pillow,
# so when pillow_simd is installed, other packages won't later override it
COPY pillow_stub /tmp/pillow_stub

RUN pip${PYTHON_VERSION} install --no-cache-dir --upgrade /tmp/pillow_stub && \
    pip${PYTHON_VERSION} install --no-cache-dir --upgrade pillow_simd==${PILLOW_SIMD_VERSION}

#################
# Install Pytorch
#################
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG CUDA_VERSION_TAG

RUN pip${PYTHON_VERSION} install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
        torch==${PYTORCH_VERSION}+${CUDA_VERSION_TAG} \
        torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION_TAG}

#########
# Cleanup
#########
RUN rm -rf /tmp/*


#######################
# Set the shell to bash
#######################
SHELL ["/bin/bash", "-c"]

################################################
# Create a virtualenv and activate it by default
################################################
RUN python${PYTHON_VERSION} -m venv --system-site-packages ~/mosaicml-venv && \
    ~/mosaicml-venv/bin/python -m pip install --upgrade pip setuptools && \
    echo "source ~/mosaicml-venv/bin/activate" >> ~/.bashrc

#########################
# Configure non-root user
#########################
RUN echo "[ ! -d ~/mosaicml-venv ] && python${PYTHON_VERSION} -m venv --system-site-packages ~/mosaicml-venv && ~/mosaicml-venv/bin/python -m pip install --upgrade pip setuptools" >> /etc/skel/.bashrc && \
    echo 'source ~/mosaicml-venv/bin/activate' >> /etc/skel/.bashrc && \
    useradd -rm -d /home/mosaicml -s /bin/bash -u 1000 -U -s /bin/bash mosaicml && \
    usermod -a -G sudo mosaicml && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/mosaicml

USER mosaicml

RUN python${PYTHON_VERSION} -m venv --system-site-packages ~/mosaicml-venv && ~/mosaicml-venv/bin/python -m pip install --upgrade pip setuptools
