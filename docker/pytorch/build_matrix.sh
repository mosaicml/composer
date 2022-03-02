#!/usr/bin/env bash
set -euo pipefail

# IMPORTANT: For gcp and A100s, the base image must be the `devel` version, not the runtime version

echo "TAG='mosaicml/pytorch:1.9.1_cu111-python3.7-ubuntu20.04' BASE_IMAGE='nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04' PYTHON_VERSION='3.7' CUDA_VERSION_TAG='cu111' CUDA_VERSION='11.1.1' LINUX_DISTRO='ubuntu:20.04' PYTORCH_VERSION='1.9.1' TORCHVISION_VERSION='0.10.1'"
echo "TAG='mosaicml/pytorch:1.9.1_cpu-python3.7-ubuntu20.04' BASE_IMAGE='ubuntu:20.04' PYTHON_VERSION='3.7' CUDA_VERSION_TAG='cpu' CUDA_VERSION='cpu' LINUX_DISTRO='ubuntu:20.04' PYTORCH_VERSION='1.9.1' TORCHVISION_VERSION='0.10.1'"
echo "TAG='mosaicml/pytorch:1.9.1_cu111-python3.8-ubuntu20.04' BASE_IMAGE='nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04' PYTHON_VERSION='3.8' CUDA_VERSION_TAG='cu111' CUDA_VERSION='11.1.1' LINUX_DISTRO='ubuntu:20.04' PYTORCH_VERSION='1.9.1' TORCHVISION_VERSION='0.10.1'"
echo "TAG='mosaicml/pytorch:1.9.1_cpu-python3.8-ubuntu20.04' BASE_IMAGE='ubuntu:20.04' PYTHON_VERSION='3.8' CUDA_VERSION_TAG='cpu' CUDA_VERSION='cpu' LINUX_DISTRO='ubuntu:20.04' PYTORCH_VERSION='1.9.1' TORCHVISION_VERSION='0.10.1'"
echo "TAG='mosaicml/pytorch:1.10.0_cu113-python3.9-ubuntu20.04' TAG='mosaicml/pytorch:latest' BASE_IMAGE='nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04' PYTHON_VERSION='3.9' CUDA_VERSION_TAG='cu113' CUDA_VERSION='11.3.1' LINUX_DISTRO='ubuntu:20.04' PYTORCH_VERSION='1.10.0' TORCHVISION_VERSION='0.11.1'"
echo "TAG='mosaicml/pytorch:1.10.0_cpu-python3.9-ubuntu20.04' BASE_IMAGE='ubuntu:20.04' PYTHON_VERSION='3.9' CUDA_VERSION_TAG='cpu' CUDA_VERSION='cpu' LINUX_DISTRO='ubuntu:20.04' PYTORCH_VERSION='1.10.0' TORCHVISION_VERSION='0.11.1'"
echo "TAG='mosaicml/pytorch:1.9.1_cu111-python3.8-ubuntu18.04' BASE_IMAGE='nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04' PYTHON_VERSION='3.8' CUDA_VERSION_TAG='cu111' CUDA_VERSION='11.1.1' LINUX_DISTRO='ubuntu:18.04' PYTORCH_VERSION='1.9.1' TORCHVISION_VERSION='0.10.1'"
echo "TAG='mosaicml/pytorch:1.9.1_cpu-python3.8-ubuntu18.04' BASE_IMAGE='ubuntu:18.04' PYTHON_VERSION='3.8' CUDA_VERSION_TAG='cpu' CUDA_VERSION='cpu' LINUX_DISTRO='ubuntu:18.04' PYTORCH_VERSION='1.9.1' TORCHVISION_VERSION='0.10.1'"
