ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

###################
# MosaicML Composer
###################

ARG COMPOSER_EXTRA_DEPS

# Install composer package
RUN pip install mosaicml[$COMPOSER_EXTRA_DEPS]

