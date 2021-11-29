ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

###################
# MosaicML Composer
###################

# Build context must be the root of the repo
COPY --chown=1000:1000 ./ /tmp/composer

ARG COMPOSER_EXTRA_DEPS

# Install composer package
RUN pip install --user /tmp/composer[$COMPOSER_EXTRA_DEPS]

RUN rm -rf /tmp/*

