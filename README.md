# BNN-PYNQ PIP INSTALL Package

This repo contains the pip install package for Binarized Neural Network (BNN) on PYNQ. 
Two different overlays are here included, namely CNV and LFC as described in the <a href="https://arxiv.org/abs/1612.07119" target="_blank"> FINN Paper </a>. 

## Citation
If you find BNN-PYNQ useful, please cite the <a href="https://arxiv.org/abs/1612.07119" target="_blank">FINN paper</a>:

    @inproceedings{finn,
    author = {Umuroglu, Yaman and Fraser, Nicholas J. and Gambardella, Giulio and Blott, Michaela and Leong, Philip and Jahre, Magnus and Vissers, Kees},
    title = {FINN: A Framework for Fast, Scalable Binarized Neural Network Inference},
    booktitle = {Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
    series = {FPGA '17},
    year = {2017},
    pages = {65--74},
    publisher = {ACM}
    }

## Quick Start

In order to install it your PYNQ, connect to the board, open a terminal and type:

sudo pip3.6 install git+https://github.com/Xilinx/BNN-PYNQ.git (on PYNQ v2.0

This will install the BNN package to your board, and create a **BNN** directory in the Jupyter home area. You will find the Jupyter notebooks to test the BNN in this directory. 

In order to build the shared object during installation, the user should copy the include folder from VIVADO HLS on the PYNQ board (in windows in vivado-path/Vivado_HLS/201x.y/include, /vivado-path/Vidado_HLS/201x.y/include in unix) and set the environment variable VIVADOHLS_INCLUDE_PATH to the location in which the folder has been copied.
If the env variable is not set, the precompiled version will be used instead. 
 
## Repo organization 

The repo is organized as follows:

-	bnn: contains the PynqBNN class description
	-	src: contains the sources of the 2 networks, the libraries to rebuild them, and scripts to train and pack the weights:
		- library: FINN library for HLS BNN descriptions, host code, script to rebuilt and drivers for the PYNQ (please refer to README for more details)
		- network: BNN topologies (CNV and LFC) HLS top functions, host code and make script for HW and SW built (please refer to README for more details)
        - training: scripts to train on the Cifar10 and MNIST datasets and scripts to pack the weights in a binary format which can be read by the BNN overlay
	-	bitstreams: with the bitstream for the 2 overlays
	-	libraries: pre-compiled shared objects for low-level driver of the 2 overlays
	-	params: set of trained parameters for the 2 networks:
		- <a href="http://yann.lecun.com/exdb/mnist/" target="_blank"> MNIST </a> and <a href="https://www.nist.gov/srd/nist-special-database-19" target="_blank"> NIST </a> dataset for LFC network
		- <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank"> Cifar10 </a>, <a href="http://ufldl.stanford.edu/housenumbers/" target="_blank"> SVHN </a> and <a href="http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset" target="_blank"> German Road Signs </a> dataset for CNV network
-	notebooks: lists a set of python notebooks examples, that during installation will be moved in `/home/xilinx/jupyter_notebooks/bnn/` folder
-	tests: contains test scripts and test images

## Hardware design rebuilt

In order to rebuild the hardware designs, the repo should be cloned in a machine with installation of the Vivado Design Suite (tested with 2016.1). 
Following the step-by-step instructions:

1.	Clone the repository on your linux machine: git clone https://github.com/Xilinx/BNN-PYNQ.git;
2.	Move to `clone_path/BNN_PYNQ/bnn/src/network/`
3.	Set the XILINX_BNN_ROOT environment variable to `clone_path/BNN_PYNQ/bnn/src/`
4.	Launch the shell script make-hw.sh with parameters the target network, target platform and mode, with the command `./make-hw.sh {network} {platform} {mode}` where:
	- network can be cnv-pynq or lfc-pynq;
	- platform is pynq;
	- mode can be `h` to launch Vivado HLS synthesis, `b` to launch the Vivado project (needs HLS synthesis results), `a` to launch both.
5.	The results will be visible in `clone_path/BNN_PYNQ/bnn/src/network/output/` that is organized as follows:
	- bitstream: contains the generated bitstream(s);
	- hls-syn: contains the Vivado HLS generated RTL and IP (in the subfolder named as the target network);
	- report: contains the Vivado and Vivado HLS reports;
	- vivado: contains the Vivado project.
6.	Copy the generated bitstream and tcl script on the PYNQ board `pip_installation_path/bnn/bitstreams/`

