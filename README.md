# BNN-PYNQ PIP INSTALL Package

This repo contains the pip install package for Quantized Neural Network (QNN) on PYNQ. 
Two different network topologies are here included, namely CNV and LFC as described in the <a href="https://arxiv.org/abs/1612.07119" target="_blank"> FINN Paper </a>. 
Now, there are multiple implementations available supporting different precision for weights and activation:

- 1 bit weights and 1 bit activation (W1A1) for CNV and LFC
- 1 bit weights and 2 bit activation (W1A2) for CNV and LFC
- 2 bit weights and 2 bit activation (W2A2) for CNV

We support 3 boards for hardware acceleration which are Pynq-Z1, Pynq-Z2 and Ultra96 (with PYNQ image).

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

Please refer to PYNQ <a href="https://pynq.readthedocs.io/en/latest/getting_started.html" target="_blank"> Getting Started</a> guide to set-up your PYNQ Board.

In order to install it to your PYNQ, connect to the board, open a terminal and type:

```
sudo pip3 install git+https://github.com/Xilinx/BNN-PYNQ.git (on PYNQ v2.3)
sudo pip3.6 install git+https://github.com/Xilinx/BNN-PYNQ.git (on PYNQ v2.2 and earlier)
```

This will install the BNN package to your board, and create a **bnn** directory in the Jupyter home area. You will find the Jupyter notebooks to test the networks in this directory.

## Repo organization 

The repo is organized as follows:

-	bnn: contains the LfcClassifier and CnvClassifier python class description
	-	src: contains the sources of the different precision networks, the libraries to rebuild them, and scripts to train and pack the weights:
		- library: FINN library for HLS QNN descriptions, host code, script to rebuilt and drivers for the PYNQ and Ultra96 (please refer to README for more details)
		- network: HLS top functions for QNN topologies (CNV and LFC) with different implementations for weight and activation precision, host code and make script for HW and SW build (please refer to README for more details)
		- training: scripts to train on the Cifar10, GTSRB and MNIST datasets and scripts to pack the weights in a binary format which can be read by the overlay
	-	bitstreams: contains the bitstreams for the 5 overlays
		- pynqZ1-Z2: bitstreams for Pynq devices
		- ultra96: bitstreams for Ultra96 devices
	-	libraries: pre-compiled shared objects for low-level driver of the 5 overlays each for hardware and software runtime
		- pynqZ1-Z2: shared objects used by Pynq devices
		- ultra96: shared objects used by ultra96
	-	params: set of trained parameters for the 5 overlays:
		- <a href="http://yann.lecun.com/exdb/mnist/" target="_blank"> MNIST </a> and <a href="https://www.nist.gov/srd/nist-special-database-19" target="_blank"> NIST </a> dataset for LFC network. Note that NIST dataset is only applicable to LFC-W1A1 by default.
		- <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank"> Cifar10 </a>, <a href="http://ufldl.stanford.edu/housenumbers/" target="_blank"> SVHN </a> and <a href="http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset" target="_blank"> German Road Signs </a> dataset for CNV network. Note that SVHN and German Road Signs databases are only applicable to CNV-W1A1 by default.
-	notebooks: lists a set of python notebooks examples, that during installation will be moved in `/home/xilinx/jupyter_notebooks/bnn/` folder
-	tests: contains test script and test images

## Hardware design rebuilt

In order to rebuild the hardware designs, the repo should be cloned in a machine with installation of the Vivado Design Suite (tested with 2018.2). 
Following the step-by-step instructions:

1.	Clone the repository on your linux machine: git clone https://github.com/Xilinx/BNN-PYNQ.git;
2.	Move to `<clone_path>/BNN_PYNQ/bnn/src/network/`
3.	Set the XILINX_BNN_ROOT environment variable to `<clone_path>/BNN_PYNQ/bnn/src/`
4.	Launch the shell script make-hw.sh with passing parameters for target network, target platform and mode, with the command `./make-hw.sh {network} {platform} {mode}` where:
	- network can be cnvW1A1, cnvW1A2, cnvW2A2 or lfcW1A1, lfcW1A2;
	- platform can be pynqZ1-Z2 or ultra96;
	- mode can be `h` to launch Vivado HLS synthesis, `b` to launch the Vivado project (needs HLS synthesis results), `a` to launch both;
5.	The results will be visible in `clone_path/BNN_PYNQ/bnn/src/network/output/` that is organized as follows:
	- bitstream: contains the generated bitstream(s);
	- hls-syn: contains the Vivado HLS generated RTL and IP (in the subfolder named as the target network and target platform);
	- report: contains the Vivado and Vivado HLS reports;
	- vivado: contains the Vivado project;
6.	Copy the generated bitstream and tcl script on the PYNQ board `pip_installation_path/bnn/bitstreams/`

