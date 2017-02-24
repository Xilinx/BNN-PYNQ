from setuptools import setup, find_packages
import subprocess
import sys
import shutil
import bnn
import os
from glob import glob

if 'BOARD' not in os.environ or os.environ['BOARD'] != 'Pynq-Z1':
    print("Only supported on a Pynq Z1 Board")
    exit(1)

setup(
    name = "bnn-pynq",
    version = bnn.__version__,
    url = 'kwa/pynq',
    license = 'Apache Software License',
    author = "People",
    author_email = "somone@somewhere.org",
    include_package_data = True,
    packages = ['bnn'],
    package_data = {
    '' : ['*.bit','*.tcl','*.so','*.bin','*.txt'],
    },
    data_files=[('/home/xilinx/jupyter_notebooks/bnn', glob('notebooks/*'))],
    description = "Classification using a hardware accelerated binary neural network"
)
