# Docker

To simplify environment setup for the MosaicML `Composer` library, we provide a set of Docker Images that users can
leverage.

## Pytorch Images

| Linux Distro | Pytorch Version | Cuda Version | Python Version | Docker Tag                                                        |
| ------------ | --------------- | ------------ | -------------- | ----------------------------------------------------------------- |
| ubuntu:20.04 | 1.11.0          | 11.3.1       | 3.9            | `mosaicml/pytorch:1.11.0_cu113-python3.9-ubuntu20.04`             |
| ubuntu:20.04 | 1.11.0          | cpu          | 3.9            | `mosaicml/pytorch:1.11.0_cpu-python3.9-ubuntu20.04`               |
| ubuntu:20.04 | 1.10.0          | 11.3.1       | 3.9            | `latest`, `mosaicml/pytorch:1.10.0_cu113-python3.9-ubuntu20.04`   |
| ubuntu:20.04 | 1.10.0          | cpu          | 3.9            | `mosaicml/pytorch:1.10.0_cpu-python3.9-ubuntu20.04`               |
| ubuntu:18.04 | 1.9.1           | 11.1.1       | 3.8            | `mosaicml/pytorch:1.9.1_cu111-python3.8-ubuntu18.04`              |
| ubuntu:18.04 | 1.9.1           | cpu          | 3.8            | `mosaicml/pytorch:1.9.1_cpu-python3.8-ubuntu18.04`                |
| ubuntu:20.04 | 1.9.1           | 11.1.1       | 3.8            | `mosaicml/pytorch:1.9.1_cu111-python3.8-ubuntu20.04`              |
| ubuntu:20.04 | 1.9.1           | cpu          | 3.8            | `mosaicml/pytorch:1.9.1_cpu-python3.8-ubuntu20.04`                |
| ubuntu:20.04 | 1.9.1           | 11.1.1       | 3.7            | `mosaicml/pytorch:1.9.1_cu111-python3.7-ubuntu20.04`              |
| ubuntu:20.04 | 1.9.1           | cpu          | 3.7            | `mosaicml/pytorch:1.9.1_cpu-python3.7-ubuntu20.04`                |

Our `latest` image has Ubuntu 20.04, Python 3.9, PyTorch 1.10, and CUDA 11.3, and has been tested to work with
GPU-based instances on AWS, GCP, and Azure. ``Pillow-SIMD`` is installed by default in all images.

**Note**: These images do not include Composer preinstalled. To install composer, once inside the image, run `pip install mosaicml`.

### Pulling Images

Pre-built images can be pulled from [MosaicML's DockerHub Repository](https://hub.docker.com/r/mosaicml/pytorch):

```bash
docker pull mosaicml/pytorch
```

## Building Images locally

```bash
# Build the default image
make

# Build with composer with Python 3.8
PYTHON_VERSION=3.8 make
```

**Note**: Docker must be [installed](https://docs.docker.com/get-docker/) on your local machine.

