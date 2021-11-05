BASE_IMAGE ?= 'nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04'
PYTHON_VERSION ?='3.9'
CUDA_VERSION_TAG ?= 'cu113'
PYTORCH_VERSION ?= '1.10.0'
TORCHVISION_VERSION ?= '0.11.1'
PYTORCH_TAG ?= mosaicml/pytorch
COMPOSER_TAG ?= mosaicml/composer
COMPOSER_EXTRA_DEPS ?= all

.PHONY: build

base:
	cd pytorch && docker build -t $(PYTORCH_TAG) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg CUDA_VERSION_TAG=$(CUDA_VERSION_TAG) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		--build-arg PYTORCH_VERSION=$(PYTORCH_VERSION) \
		--build-arg TORCHVISION_VERSION=$(TORCHVISION_VERSION) \
		.

composer: base
	cd .. && docker build -t $(COMPOSER_TAG) \
		--build-arg BASE_IMAGE=$(PYTORCH_TAG) \
		--build-arg COMPOSER_EXTRA_DEPS=$(COMPOSER_EXTRA_DEPS) \
		-f docker/composer/Dockerfile \
		.

build: composer
