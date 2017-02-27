# BNN-PYNQ PIP INSTALL Package

This repo contains the pip install package for Binarized Neural Network (BNN) on PYNQ. 
Two different overlays are here included, namely CNV and LFC as described in the <a href="https://arxiv.org/abs/1612.07119" target="_blank"> FINN Paper </a>. 

## Quick Start

In order to install it your PYNQ, connect to the board, open a terminal and type:

sudo pip3.6 install git+https://github.com/Xilinx/BNN-PYNQ.git (on PYNQ v1.4)

sudo pip3.4 install git+https://github.com/Xilinx/BNN-PYNQ.git (on PYNQ v1.3)

## Repo organization 

The repo is organized as follows:

-	bnn: contains the PynqBNN class description
	-	bitstreams: with the bitstream for the 2 overlays
	-	libraries: pre-compiled shared objects for low-level driver of the 2 overlays
	-	params: set of trained parameters for the 2 networks:
		- <a href="http://yann.lecun.com/exdb/mnist/" target="_blank"> MNIST </a> dataset for LFC network
		- <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank"> Cifar10 </a>, <a href="http://ufldl.stanford.edu/housenumbers/" target="_blank"> SVHN </a> and <a href="http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset" target="_blank"> German Road Signs </a> dataset for CNV network
-	notebooks: lists a set of python notebooks examples, that during installation will be moved in `/home/xilinx/jupyter_notebooks/bnn/` folder
-	tests: contains test scripts and test images





