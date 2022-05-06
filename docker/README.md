# Docker

To simplify environment setup for the MosaicML `Composer` library, we provide a set of Docker Images that users can
leverage.

## Pytorch Images

| Linux Distro   | Flavor   | PyTorch Version   | CUDA Version   | Python Version   | Docker Tags                                                                                       |
|----------------|----------|-------------------|----------------|------------------|---------------------------------------------------------------------------------------------------|
| Ubuntu 20.04   | Base     | 1.11.0            | 11.3.1         | 3.10             | `mosaicml/pytorch:latest`, `mosaicml/pytorch:1.11.0_cu113-python3.10-ubuntu20.04`                 |
| Ubuntu 20.04   | Base     | 1.11.0            | cpu            | 3.10             | `mosaicml/pytorch:latest_cpu`, `mosaicml/pytorch:1.11.0_cpu-python3.10-ubuntu20.04`               |
| Ubuntu 20.04   | Base     | 1.10.2            | 11.3.1         | 3.9              | `mosaicml/pytorch:1.10.2_cu113-python3.9-ubuntu20.04`                                             |
| Ubuntu 20.04   | Base     | 1.10.2            | cpu            | 3.9              | `mosaicml/pytorch:1.10.2_cpu-python3.9-ubuntu20.04`                                               |
| Ubuntu 20.04   | Base     | 1.10.2            | 11.3.1         | 3.8              | `mosaicml/pytorch:1.10.2_cu113-python3.8-ubuntu20.04`                                             |
| Ubuntu 20.04   | Base     | 1.10.2            | cpu            | 3.8              | `mosaicml/pytorch:1.10.2_cpu-python3.8-ubuntu20.04`                                               |
| Ubuntu 20.04   | Base     | 1.10.2            | 11.3.1         | 3.7              | `mosaicml/pytorch:1.10.2_cu113-python3.7-ubuntu20.04`                                             |
| Ubuntu 20.04   | Base     | 1.10.2            | cpu            | 3.7              | `mosaicml/pytorch:1.10.2_cpu-python3.7-ubuntu20.04`                                               |
| Ubuntu 20.04   | Vision   | 1.11.0            | 11.3.1         | 3.10             | `mosaicml/pytorch_vision:latest`, `mosaicml/pytorch_vision:1.11.0_cu113-python3.10-ubuntu20.04`   |
| Ubuntu 20.04   | Vision   | 1.11.0            | cpu            | 3.10             | `mosaicml/pytorch_vision:latest_cpu`, `mosaicml/pytorch_vision:1.11.0_cpu-python3.10-ubuntu20.04` |
| Ubuntu 20.04   | Vision   | 1.10.2            | 11.3.1         | 3.9              | `mosaicml/pytorch_vision:1.10.2_cu113-python3.9-ubuntu20.04`                                      |
| Ubuntu 20.04   | Vision   | 1.10.2            | cpu            | 3.9              | `mosaicml/pytorch_vision:1.10.2_cpu-python3.9-ubuntu20.04`                                        |

``Pillow-SIMD`` is installed by default in all images.

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

