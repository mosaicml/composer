FROM mosaicml/python:latest

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
                torch==1.9.0+cu111 \
                torchvision==0.10.0+cu111
