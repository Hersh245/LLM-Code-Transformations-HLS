#!/bin/bash

if [ "$ID" == "ubuntu" ]; then
    source /tools/Xilinx/Vitis_HLS/2023.1/settings64.sh 
    source /opt/xilinx/xrt/setup.sh
    source /opt/merlin/sources/merlin-compiler/merlin_setting.sh
    export C_INCLUDE_PATH="/tools/Xilinx/Vitis_HLS/2023.1/include:$C_INCLUDE_PATH"
    export CPLUS_INCLUDE_PATH="/tools/Xilinx/Vitis_HLS/2023.1/include:$CPLUS_INCLUDE_PATH"
elif [ "$ID" == "centos" ]; then
    if [ ! -d $AWS_FPGA_REPO_DIR ]; then
        git clone https://github.com/aws/aws-fpga.git $AWS_FPGA_REPO_DIR
    fi
    pushd $AWS_FPGA_REPO_DIR
        source vitis_setup.sh
    popd

    source /opt/Xilinx/Vitis_HLS/2021.2/settings64.sh
    source /opt/xilinx/xrt/setup.sh
    source /opt/merlin/sources/merlin-compiler/merlin_setting.sh
    export C_INCLUDE_PATH="/opt/Xilinx/Vitis_HLS/2021.2/include:$C_INCLUDE_PATH"
    export CPLUS_INCLUDE_PATH="/opt/Xilinx/Vitis_HLS/2021.2/include:$CPLUS_INCLUDE_PATH"
else
    echo "Unsupported distribution: $ID"
    exit 1
fi