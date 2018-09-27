#!/bin/bash
###############################################################################
 #  Copyright (c) 2016, Xilinx, Inc.
 #  All rights reserved.
 #
 #  Redistribution and use in source and binary forms, with or without
 #  modification, are permitted provided that the following conditions are met:
 #
 #  1.  Redistributions of source code must retain the above copyright notice,
 #     this list of conditions and the following disclaimer.
 #
 #  2.  Redistributions in binary form must reproduce the above copyright
 #      notice, this list of conditions and the following disclaimer in the
 #      documentation and/or other materials provided with the distribution.
 #
 #  3.  Neither the name of the copyright holder nor the names of its
 #      contributors may be used to endorse or promote products derived from
 #      this software without specific prior written permission.
 #
 #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 #  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 #  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 #  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 #  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 #  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 #  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 #  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 #  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 #  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 #  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
###############################################################################
###############################################################################
 #
 #
 # @file make-sw.sh
 #
 # Bash script for host code compiling into a shared object used by the BNN 
 # python class. This script should be launched on the PYNQ board. 
 #
 #
###############################################################################

NETWORKS=$(ls -d *W*A*/ | cut -f1 -d'/' | tr "\n" " ")

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <network> <runtime> where ">&2
  echo "<network> = $NETWORKS" >&2
  echo "<runtime> = python_sw python_hw" >&2
  exit 1
fi

NETWORK=$1
RUNTIME=$2

if [ -z "$XILINX_BNN_ROOT" ]; then
  echo "Need to set XILINX_BNN_ROOT"
  exit 1
fi

if [ -z "$VIVADOHLS_INCLUDE_PATH" ]; then
  echo "Need to set VIVADOHLS_INCLUDE_PATH to rebuild from source"
  echo "The pre-compiled shared objects will be included"
  exit 1
fi  

OLD_DIR=$(pwd)
cd $XILINX_BNN_ROOT
if [ -d "xilinx-tiny-cnn/" ]; then
  echo "xilinx-tiny-cnn already cloned"
else
  git clone https://github.com/Xilinx/xilinx-tiny-cnn.git
fi
cd $OLD_DIR

if [[ ("$BOARD" == "Pynq-Z1") || ("$BOARD" == "Pynq-Z2") ]]; then
  DEF_BOARD="PYNQ"
  PLATFORM="pynqZ1-Z2"  
elif [[ ("$BOARD" == "Ultra96") ]]; then
  DEF_BOARD="ULTRA"
  PLATFORM="ultra96"
else
  echo "Error: BOARD variable has to be Ultra96, Pynq-Z1 and Pynq-Z2 Board."
  exit 1
fi

TINYCNN_PATH=$XILINX_BNN_ROOT/xilinx-tiny-cnn
BNN_PATH=$XILINX_BNN_ROOT/network
BNNLIB=$XILINX_BNN_ROOT/library
HOSTLIB=$BNNLIB/host
HLSLIB=$BNNLIB/hls
HLSTOP=$BNN_PATH/$NETWORK/hw
DRIVER_PATH=$BNNLIB/driver

SRCS_HOSTLIB=$HOSTLIB/*.cpp
SRCS_HLSLIB=$HLSLIB/*.cpp
SRCS_HLSTOP=$HLSTOP/top.cpp
SRCS_HOST=$BNN_PATH/$NETWORK/sw/main.cpp

OUTPUT_DIR=$XILINX_BNN_ROOT/network/output/sw
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/$RUNTIME-$NETWORK-$PLATFORM"


if [[ ("$RUNTIME" == "python_sw") ]]; then
  SRCS_HOST=$BNN_PATH/$NETWORK/sw/main_python.cpp
  SRCS_ALL="$SRCS_HOSTLIB $SRCS_HLSTOP $SRCS_HOST"
  g++ -g -DOFFLOAD -DRAWHLS -std=c++11 -pthread -O2 -fPIC -shared $SRCS_ALL -I$VIVADOHLS_INCLUDE_PATH -I$TINYCNN_PATH -I$HOSTLIB -I$HLSLIB -I$HLSTOP -o $OUTPUT_FILE.so
elif [[ ("$RUNTIME" == "python_hw") ]]; then
  SRCS_HOST=$BNN_PATH/$NETWORK/sw/main_python.cpp
  SRCS_ALL="$DRIVER_PATH/platform-xlnk.cpp $SRCS_HOSTLIB $SRCS_HOST"
  g++ -g -DOFFLOAD -D$DEF_BOARD -std=c++11 -pthread -O3 -fPIC -shared $SRCS_ALL -I$DRIVER_PATH -I$VIVADOHLS_INCLUDE_PATH -I$TINYCNN_PATH -I$HOSTLIB -I$HLSLIB -I$HLSTOP -o $OUTPUT_FILE.so -lcma
fi

echo "Output at $OUTPUT_FILE"
