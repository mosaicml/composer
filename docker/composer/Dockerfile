ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

###################
# MosaicML Composer
###################

# Build context must be the root of the repo
COPY . /tmp/composer

ARG COMPOSER_EXTRA_DEPS

# Install composer package
RUN pip install --user src/composer[$COMPOSER_EXTRA_DEPS]
