#!/bin/bash

if [ -f "/etc/os-release" ]; then
    . /etc/os-release

    if [ "$ID" == "ubuntu" ]; then
        sudo docker run \
            -w $PWD \
            -e ID=$ID \
            -v /home:/home \
            -v /opt/xilinx:/opt/xilinx \
            -v /tools:/tools \
            -it ghcr.io/ucla-vast/merlin-ucla
    elif [ "$ID" == "centos" ]; then
        sudo systemctl start docker
        sudo docker run \
            -w $PWD \
            -e ID=$ID \
            -e AWS_FPGA_REPO_DIR=$AWS_FPGA_REPO_DIR \
            -e XILINX_VITIS=$XILINX_VITIS \
            -e XILINX_HLS=$XILINX_HLS \
            -v /home:/home \
            -v /opt/xilinx:/opt/xilinx \
            -v /opt/Xilinx:/opt/Xilinx \
            -v /etc/OpenCL:/etc/OpenCL \
            -it ghcr.io/ucla-vast/merlin-ucla
    else
        echo "Unsupported distribution: $ID"
        exit 1
    fi
else
    echo "Unable to determine the distribution."
    exit 1
fi