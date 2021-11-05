BASE_IMAGE ?= 'nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04'
PYTHON_VERSION ?='3.9'
PYTORCH_VERSION ?= '1.10.0+cu113'
TORCHVISION_VERSION ?= '0.11.1+cu113'
PYTORCH_TAG ?= mosaicml/pytorch
COMPOSER_TAG ?= mosaicml/composer

.PHONY: build

base:
	cd pytorch
	docker build -t $(PYTORCH_TAG) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		--build-arg PYTORCH_VERSION=$(PYTORCH_VERSION) \
		--build-arg TORCHVISION_VERSION=$(TORCHVISION_VERSION) \
		.

composer: base
	cd composer
	docker build -t $(COMPOSER_TAG) \
		--build-arg BASE_IMAGE=$(PYTORCH_TAG) \
		.

build: composer
