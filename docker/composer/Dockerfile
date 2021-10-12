FROM mosaicml/pytorch:latest

ARG build_option

RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /workspace

COPY . ./src/composer

# Install composer package
RUN pip install -e src/composer[$build_option]

# Install accelerated Pillow-SIMD package, default pillow must first be uninstalled per instructions:
# https://github.com/uploadcare/pillow-simd
RUN pip uninstall -y pillow && \
    pip install -U pillow_simd==6.2.2.post1 --no-cache-dir

