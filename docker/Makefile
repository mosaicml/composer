IMAGE_NAME ?= mosaicml/composer
BUILD_TYPE ?= base

ifeq (${BUILD_TYPE}, base)
	# Base dependency build tagged with 'latest'
	IMAGE_TAG = latest
else
	# Otherwise just use the BUILD_TYPE
	IMAGE_TAG = ${BUILD_TYPE}
endif

.PHONY: build

build:
	docker build .. -t $(IMAGE_NAME):${IMAGE_TAG} -f composer/Dockerfile --build-arg COMPOSER_LIB_DEP=${BUILD_TYPE}

