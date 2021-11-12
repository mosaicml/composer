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
