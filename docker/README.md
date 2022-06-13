# Docker

To simplify environment setup for the MosaicML `Composer` library, we provide a set of Docker Images that users can
leverage.

## Composer Images

The [mosaicml/composer](https://hub.docker.com/r/mosaicml/composer) images contain all Composer pre-installed with all dependencies for both NLP and Vision models. They are built on top of the [`mosaicml/pytorch_vision`](https://hub.docker.com/r/mosaicml/pytorch_vision) family of images. (See the section on [MosaicML PyTorch Images](#pytorch-images) below.)

**Note**: Only the Dockerimage for most recent version of Composer will be maintained. We recommend using `mosaicml/composer:latest` or `mosaicml/composer:latest_cpu`, which will always be up to date. Out-of-date images
may be available on DockerHub, but are not shown in the table below.

<!-- BEGIN_COMPOSER_BUILD_MATRIX -->
| Composer Version   | CUDA Support   | Docker Tag                     |
|--------------------|----------------|--------------------------------|
| latest             | Yes            | `mosaicml/composer:latest`     |
| latest             | No             | `mosaicml/composer:latest_cpu` |
| 0.7.1              | Yes            | `mosaicml/composer:0.7.1`      |
| 0.7.1              | No             | `mosaicml/composer:0.7.1_cpu`  |
<!-- END_COMPOSER_BUILD_MATRIX -->


**Note**: For a lightweight installation, we recommended using a [MosaicML PyTorch Image](#pytorch-images) and manually installing Composer within the image.

## PyTorch Images

The [mosaicml/pytorch](https://hub.docker.com/r/mosaicml/pytorch) images images contain PyTorch preinstalled, without Composer. The `Base` flavor of Docker Images contains PyTorch pre-installed; the `Vision` flavor also includes OpenCV,
MM Segmentation, and FFCV dependencies. To install composer, once inside the image, run `pip install mosaicml`.

<!-- BEGIN_PYTORCH_BUILD_MATRIX -->
| Linux Distro   | Flavor   | PyTorch Version   | CUDA Version   | Python Version   | Docker Tags                                                                                      |
|----------------|----------|-------------------|----------------|------------------|--------------------------------------------------------------------------------------------------|
| Ubuntu 20.04   | Base     | 1.11.0            | 11.3.1         | 3.9              | `mosaicml/pytorch:latest`, `mosaicml/pytorch:1.11.0_cu113-python3.9-ubuntu20.04`                 |
| Ubuntu 20.04   | Base     | 1.11.0            | cpu            | 3.9              | `mosaicml/pytorch:latest_cpu`, `mosaicml/pytorch:1.11.0_cpu-python3.9-ubuntu20.04`               |
| Ubuntu 20.04   | Base     | 1.10.2            | 11.3.1         | 3.8              | `mosaicml/pytorch:1.10.2_cu113-python3.8-ubuntu20.04`                                            |
| Ubuntu 20.04   | Base     | 1.10.2            | cpu            | 3.8              | `mosaicml/pytorch:1.10.2_cpu-python3.8-ubuntu20.04`                                              |
| Ubuntu 20.04   | Base     | 1.9.1             | 11.1.1         | 3.7              | `mosaicml/pytorch:1.9.1_cu111-python3.7-ubuntu20.04`                                             |
| Ubuntu 20.04   | Base     | 1.9.1             | cpu            | 3.7              | `mosaicml/pytorch:1.9.1_cpu-python3.7-ubuntu20.04`                                               |
| Ubuntu 20.04   | Vision   | 1.11.0            | 11.3.1         | 3.9              | `mosaicml/pytorch_vision:latest`, `mosaicml/pytorch_vision:1.11.0_cu113-python3.9-ubuntu20.04`   |
| Ubuntu 20.04   | Vision   | 1.11.0            | cpu            | 3.9              | `mosaicml/pytorch_vision:latest_cpu`, `mosaicml/pytorch_vision:1.11.0_cpu-python3.9-ubuntu20.04` |
<!-- END_PYTORCH_BUILD_MATRIX -->

``Pillow-SIMD`` is installed by default in all images.

### Pulling Images

Pre-built images can be pulled from [MosaicML's DockerHub Repository](https://hub.docker.com/u/mosaicml):

<!--pytest-codeblocks:skip-->
```bash
docker pull mosaicml/composer
```
