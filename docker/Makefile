# Base image to use for the docker build.
BASE_IMAGE ?= 'nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04'

# Python version to install. Composer requires python>=3.7
PYTHON_VERSION ?='3.9'

# Pytorch version to install. Composer requires pytorch>=1.9
PYTORCH_VERSION ?= '1.10.0'

# Torchvision version to install.
# See https://github.com/pytorch/vision#installation
# to determine the correct torchvision version for the
# corresponding pytorch version.
TORCHVISION_VERSION ?= '0.11.1'

# Flavor of pytorch and torchvision to install.
# Should correspond to the cuda version in the $BASE_IMAGE.
# Pytorch and torchvision are installed like:
# pip install torch==${PYTORCH_VERSION}+${CUDA_VERSION_TAG}
# pip install torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION_TAG}
CUDA_VERSION_TAG ?= 'cu113'

# Resulting docker tag for an image containing python, pytorch, and torchvision.
# Useful for running tests on a development version of composer.
PYTORCH_TAG ?= mosaicml/pytorch

# Extra dependencies to install for composer.
# Composer is installed like:
# pip install composer[$COMPOSER_EXTRA_DEPS]
COMPOSER_EXTRA_DEPS ?= base

# Resulting docker tag for the composer dockerimage
COMPOSER_TAG ?= mosaicml/composer:$(COMPOSER_EXTRA_DEPS)

# target stage to build
TARGET ?= pytorch_stage

MMCV_TORCH_VERSION ?= 'torch1.9.0'

.PHONY: build pytorch composer

build: composer

pytorch:
	cd pytorch && docker build -t $(PYTORCH_TAG) \
		--target $(TARGET) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg CUDA_VERSION_TAG=$(CUDA_VERSION_TAG) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		--build-arg PYTORCH_VERSION=$(PYTORCH_VERSION) \
		--build-arg TORCHVISION_VERSION=$(TORCHVISION_VERSION) \
		--build-arg MMCV_TORCH_VERSION=$(MMCV_TORCH_VERSION) \
		.

composer: pytorch
	docker build -t $(COMPOSER_TAG) \
		--build-arg BASE_IMAGE=$(PYTORCH_TAG) \
		--build-arg COMPOSER_EXTRA_DEPS=$(COMPOSER_EXTRA_DEPS) \
		-f composer/Dockerfile \
		.
