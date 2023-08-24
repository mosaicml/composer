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
| Composer Version   | CUDA Support   | Docker Tag                                                     |
|--------------------|----------------|----------------------------------------------------------------|
| 0.16.0             | Yes            | `mosaicml/composer:latest`, `mosaicml/composer:0.16.0`         |
| 0.16.0             | No             | `mosaicml/composer:latest_cpu`, `mosaicml/composer:0.16.0_cpu` |
<!-- END_COMPOSER_BUILD_MATRIX -->

**Note**: For a lightweight installation, we recommended using a [MosaicML PyTorch Image](#pytorch-images) and manually
installing Composer within the image.

## PyTorch Images

The [`mosaicml/pytorch`](https://hub.docker.com/r/mosaicml/pytorch) images contain PyTorch preinstalled, without Composer.
The base flavor contains PyTorch pre-installed; the vision flavor also includes OpenCV, MM Segmentation, and FFCV dependencies.
To install composer, once inside the image, run `pip install mosaicml`.

<!-- BEGIN_PYTORCH_BUILD_MATRIX -->
| Linux Distro   | Flavor   | PyTorch Version   | CUDA Version        | Python Version   | Docker Tags                                                                                       |
|----------------|----------|-------------------|---------------------|------------------|---------------------------------------------------------------------------------------------------|
| Ubuntu 20.04   | Base     | 2.1.0             | 12.1.0 (Infiniband) | 3.10             | `mosaicml/pytorch_nightly:2.1.0_cu121-python3.10-ubuntu20.04`                                     |
| Ubuntu 20.04   | Base     | 2.0.1             | 11.8.0 (Infiniband) | 3.10             | `mosaicml/pytorch:2.0.1_cu118-python3.10-ubuntu20.04`                                             |
| Ubuntu 20.04   | Base     | 2.0.1             | 11.8.0 (EFA)        | 3.10             | `mosaicml/pytorch:2.0.1_cu118-python3.10-ubuntu20.04-aws`                                         |
| Ubuntu 20.04   | Base     | 2.0.1             | cpu                 | 3.10             | `mosaicml/pytorch:2.0.1_cpu-python3.10-ubuntu20.04`                                               |
| Ubuntu 20.04   | Base     | 1.13.1            | 11.7.1 (Infiniband) | 3.10             | `mosaicml/pytorch:latest`, `mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04`                 |
| Ubuntu 20.04   | Base     | 1.13.1            | 11.7.1 (EFA)        | 3.10             | `mosaicml/pytorch:latest-aws`, `mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04-aws`         |
| Ubuntu 20.04   | Base     | 1.13.1            | cpu                 | 3.10             | `mosaicml/pytorch:latest_cpu`, `mosaicml/pytorch:1.13.1_cpu-python3.10-ubuntu20.04`               |
| Ubuntu 20.04   | Vision   | 2.0.1             | 11.8.0 (Infiniband) | 3.10             | `mosaicml/pytorch_vision:2.0.1_cu118-python3.10-ubuntu20.04`                                      |
| Ubuntu 20.04   | Vision   | 2.0.1             | cpu                 | 3.10             | `mosaicml/pytorch_vision:2.0.1_cpu-python3.10-ubuntu20.04`                                        |
| Ubuntu 20.04   | Vision   | 1.13.1            | 11.7.1 (Infiniband) | 3.10             | `mosaicml/pytorch_vision:latest`, `mosaicml/pytorch_vision:1.13.1_cu117-python3.10-ubuntu20.04`   |
| Ubuntu 20.04   | Vision   | 1.13.1            | cpu                 | 3.10             | `mosaicml/pytorch_vision:latest_cpu`, `mosaicml/pytorch_vision:1.13.1_cpu-python3.10-ubuntu20.04` |
<!-- END_PYTORCH_BUILD_MATRIX -->

**Note**: The `mosaicml/pytorch:latest`, `mosaicml/pytorch:latest_cpu`,`mosaicml/pytorch_vision:latest` and `mosaicml/pytorch_vision:latest_cpu`
images will always point to the stable version of PyTorch which we have battle tested and recommend for production use.  The `latest` label
may not point to an image with the most recent release of PyTorch, however we do update our images frequently so that newer versions can
be proven out.

**Note**: Only the images listed in the table above are maintained.  All other images in the DockerHub repository have been deprecated
and are kept for legacy support.  Legacy images might be cleaned up without notice so it's best to migrate to the latest image or re-tag and maintain
a private copy if you wish to continue using legacy images.

## Pulling Images

Pre-built images can be pulled from [MosaicML's DockerHub Repository](https://hub.docker.com/u/mosaicml).

For example:

<!--pytest.mark.skip-->
```bash
docker pull mosaicml/composer
```
