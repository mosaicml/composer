# Docker

To simplify environment setup for Composer, we provide a set of pre-built Docker images.

## Composer Images

The [`mosaicml/composer`](https://hub.docker.com/r/mosaicml/composer) images contain all Composer pre-installed with
all dependencies for both NLP and Vision models. They are built on top of the
[`mosaicml/pytorch_vision`](https://hub.docker.com/r/mosaicml/pytorch_vision) family of images.
(See the section on [MosaicML PyTorch Images](#pytorch-images) below.)

**Note**: Only the Dockerimage for most recent version of Composer will be maintained. We recommend using
`mosaicml/composer:latest` or `mosaicml/composer:latest_cpu`, which will always be up to date.

<!-- BEGIN_COMPOSER_BUILD_MATRIX -->
| Composer Version   | CUDA Support   | Docker Tag                     |
|--------------------|----------------|--------------------------------|
| latest             | Yes            | `mosaicml/composer:latest`     |
| latest             | No             | `mosaicml/composer:latest_cpu` |
| 0.8.2              | Yes            | `mosaicml/composer:0.8.2`      |
| 0.8.2              | No             | `mosaicml/composer:0.8.2_cpu`  |
<!-- END_COMPOSER_BUILD_MATRIX -->


**Note**: For a lightweight installation, we recommended using a [MosaicML PyTorch Image](#pytorch-images) and manually
installing Composer within the image.

## PyTorch Images

The [`mosaicml/pytorch`](https://hub.docker.com/r/mosaicml/pytorch) images contain PyTorch preinstalled, without Composer.
The base flavor contains PyTorch pre-installed; the vision flavor also includes OpenCV, MM Segmentation, and FFCV dependencies.
To install composer, once inside the image, run `pip install mosaicml`.

<!-- BEGIN_PYTORCH_BUILD_MATRIX -->
| Linux Distro   | Flavor   | PyTorch Version   | CUDA Version   | Python Version   | Docker Tags                                                                                      |
|----------------|----------|-------------------|----------------|------------------|--------------------------------------------------------------------------------------------------|
| Ubuntu 20.04   | Base     | 1.12.0            | 11.6.2         | 3.9              | `mosaicml/pytorch:latest`, `mosaicml/pytorch:1.12.0_cu116-python3.9-ubuntu20.04`                 |
| Ubuntu 20.04   | Base     | 1.12.0            | cpu            | 3.9              | `mosaicml/pytorch:latest_cpu`, `mosaicml/pytorch:1.12.0_cpu-python3.9-ubuntu20.04`               |
| Ubuntu 20.04   | Base     | 1.11.0            | 11.5.2         | 3.8              | `mosaicml/pytorch:1.11.0_cu115-python3.8-ubuntu20.04`                                            |
| Ubuntu 20.04   | Base     | 1.11.0            | cpu            | 3.8              | `mosaicml/pytorch:1.11.0_cpu-python3.8-ubuntu20.04`                                              |
| Ubuntu 20.04   | Base     | 1.10.2            | 11.3.1         | 3.7              | `mosaicml/pytorch:1.10.2_cu113-python3.7-ubuntu20.04`                                            |
| Ubuntu 20.04   | Base     | 1.10.2            | cpu            | 3.7              | `mosaicml/pytorch:1.10.2_cpu-python3.7-ubuntu20.04`                                              |
| Ubuntu 20.04   | Vision   | 1.12.0            | 11.6.2         | 3.9              | `mosaicml/pytorch_vision:latest`, `mosaicml/pytorch_vision:1.12.0_cu116-python3.9-ubuntu20.04`   |
| Ubuntu 20.04   | Vision   | 1.12.0            | cpu            | 3.9              | `mosaicml/pytorch_vision:latest_cpu`, `mosaicml/pytorch_vision:1.12.0_cpu-python3.9-ubuntu20.04` |
<!-- END_PYTORCH_BUILD_MATRIX -->

``Pillow-SIMD`` is installed by default in all images.

## Pulling Images

Pre-built images can be pulled from [MosaicML's DockerHub Repository](https://hub.docker.com/u/mosaicml).

For example:

<!--pytest.mark.skip-->
```bash
docker pull mosaicml/composer
```
