#!/bin/bash

# Same as aws_ml_env_setup.sh but without python and pip installation as that is managed by poetry.


# Check if CUDA 12.4 is installed
if [ ! -d "/usr/local/cuda-12.6" ]; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
    sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-6
else
    echo "CUDA 12.6 is already installed."
fi

# Check if a specific NVIDIA driver is installed
NVIDIA_DRIVER_VERSION="nvidia-driver-550"
if ! dpkg -l | grep -q "$NVIDIA_DRIVER_VERSION"; then
    echo "$NVIDIA_DRIVER_VERSION is not installed. Installing now..."
    # Update package list and install the NVIDIA driver
    sudo apt update
    sudo apt install -y "$NVIDIA_DRIVER_VERSION"
else
    echo "$NVIDIA_DRIVER_VERSION is already installed."
fi

# Reload NVIDIA driver module
if ! lsmod | grep -q "nvidia"; then
    echo "Loading NVIDIA driver module..."
    sudo modprobe nvidia
else
    echo "NVIDIA driver module is already loaded."
fi

# Check if cuDNN is installed
CUDNN_FILE="/usr/include/cudnn_version.h"
if [ -f "$CUDNN_FILE" ]; then
    echo "cuDNN is already installed."
    grep "CUDNN_MAJOR" "$CUDNN_FILE" -A 2
else
    echo "cuDNN is not installed. Installing now..."
    # Add NVIDIA repository for cuDNN
    wget https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/cudnn-local-repo-ubuntu2404-9.6.0_1.0-1_amd64.deb
    sudo dpkg -i cudnn-local-repo-ubuntu2404-9.6.0_1.0-1_amd64.deb
    sudo cp /var/cudnn-local-repo-ubuntu2404-9.6.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    # Install cuDNN
    sudo apt-get -y install cudnn-cuda-12
fi
