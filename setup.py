#   Copyright (c) 2016, Xilinx, Inc.
#   All rights reserved.
# 
#   Redistribution and use in source and binary forms, with or without 
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright 
#       notice, this list of conditions and the following disclaimer in the 
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its 
#       contributors may be used to endorse or promote products derived from 
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from setuptools import setup, find_packages
import subprocess
import sys
import shutil
import bnn
import os
from glob import glob
import site 

if 'BOARD' not in os.environ or os.environ['BOARD'] != 'Pynq-Z1':
    print("Only supported on a Pynq Z1 Board")
    exit(1)



setup(
    name = "bnn-pynq",
    version = bnn.__version__,
    url = 'kwa/pynq',
    license = 'Apache Software License',
    author = "Nicholas Fraser, Giulio Gambardella, Peter Ogden, Yaman Umuroglu",
    author_email = "pynq_support@xilinx.com",
    include_package_data = True,
    packages = ['bnn'],
    package_data = {
    '' : ['*.bit','*.tcl','*.so','*.bin','*.txt', '*.cpp', '*.h', '*.sh'],
    },    
    data_files = [(os.path.join('/home/xilinx/jupyter_notebooks/bnn',root.replace('notebooks/','')), [os.path.join(root, f) for f in files]) for root, dirs, files in os.walk('notebooks/')],
    #data_files = [(os.path.join('/home/xilinx/jupyter_notebooks/bnn',root.replace('notebooks/','')), [os.path.join(root, f) for f in files]) for root, dirs, files in os.walk('bnn/src/')],
    description = "Classification using a hardware accelerated binary neural network"
   
)


def run_make(src_path, network, output_type):
    status = subprocess.check_call(["bash", src_path + "/make-sw.sh", network, output_type])
    if status is not 0:
        print("Error while running make for",network,output_type,"Exiting..")
        exit(1)
    shutil.copyfile( src_path + "/output/sw/" + output_type + "-" + network + ".so", src_path + "../../libraries/" +  output_type + "-" + network + ".so")

if len(sys.argv) > 1 and sys.argv[1] == 'install' and 'VIVADOHLS_INCLUDE_PATH' in os.environ:
   os.environ["XILINX_BNN_ROOT"] = site.getsitepackages()[0] + "/bnn/src/"
   XILINX_BNN_ROOT=site.getsitepackages()[0]
   #BNN_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
   run_make(XILINX_BNN_ROOT+"/bnn/src/network/", "cnv-pynq" ,"python_sw")
   run_make(XILINX_BNN_ROOT+"/bnn/src/network/", "lfc-pynq" ,"python_sw")
   run_make(XILINX_BNN_ROOT+"/bnn/src/network/", "cnv-pynq" ,"python_hw")
   run_make(XILINX_BNN_ROOT+"/bnn/src/network/", "lfc-pynq" ,"python_hw")
else:
  print("VIVADOHLS_INCLUDE_PATH variable not set, the source will not be recompiled.",file=sys.stdout)
