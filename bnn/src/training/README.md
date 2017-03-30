# BinaryNets for Pynq

## Motivations

This repository trains the MNIST and CIFAR-10 networks provided with the FINN Pynq overlay. 
The overlay can be found here: [https://github.com/Xilinx/BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ)

## Requirements

* Python 2.7, Numpy (version 1.10.x or 1.11.x), Scipy
* [Theano](http://deeplearning.net/software/theano/install.html) (version 0.9.0beta1)
* [Pylearn2](http://deeplearning.net/software/pylearn2/)
* [Download the datasets](#download-datasets) you need
* [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html) (version 0.2.dev1)

### Recommended

* A fast Nvidia GPU (or a large amount of patience)
* Set your [Theano flags](http://deeplearning.net/software/theano/library/config.html) to use the GPU
* If using a non-dedicated system, use [virtualenv](https://virtualenv.pypa.io/) to install the python requirements
* If you don't have access to a GPU, use an [Amazon Web Service EC2 container](#installing-the-training-environment)

## Training Networks

### MNIST MLP

```bash
    $ python mnist.py
```
    
This python script trains an MLP (denoted LFC) on MNIST with BinaryNet.
It should run for about 2 hours on a GRID K520 GPU (i.e., a g2.2xlarge instance on AWS.)
The final test error should be around **1.60%**.

### CIFAR-10 ConvNet

```bash
    $ python cifar10.py
```
    
This python script trains a ConvNet (denoted CNV) on CIFAR-10 with BinaryNet.
It should run for about 43 hours on a GRID K520 GPU (i.e., a g2.2xlarge instance on AWS.)
With cuDNN installed, it should be about 12 hours.
The final test error should be around **20.42%**.

### Training Your Own Networks

The inputs to the training scripts are numpy arrays.
In order to train an LFC or a CNV network see the steps below.

#### CNV:
1. Import training data and store in a Nx3x32x32 numpy array, with each value scaled from -1 to 1.
1. Put each output label into [one hot encoded](https://en.wikipedia.org/wiki/One-hot) format.
1. Break the inputs and outputs up into training, validation and test sets.
1. Replace the lines in cifar10.py which load the training, validation and test sets with your newly imported and formatted numpy arrays.
1. Train the network, if sufficient accuracy isn't achieved, modify some training parameters and try again.

#### LFC:
1. Import training data and store in a Nx1x28x28 numpy array, with each value either -1 to 1.
1. Put each output label into [one hot encoded](https://en.wikipedia.org/wiki/One-hot) format.
1. Break the inputs and outputs up into training, validation and test sets.
1. Replace the lines in mnist.py which load the training, validation and test sets with your newly imported and formatted numpy arrays.
1. Train the network, if sufficient accuracy isn't achieved, modify some training parameters and try again.

## Generating Binary Weights Files

Once the training process has finished, you'll have a file DATASET_parameters.npz (where DATASET is either "mnist" or "cifar10") containing a list of numpy arrays which correspond to the real trained weights in each layer.
In order to load them into the [Pynq BNN Overlay](https://github.com/Xilinx/BNN-PYNQ) they need converted from real floating point values into binary values and packed into .bin files. 

```bash
    $ python DATASET-gen-binary-weights.py
```

These scripts will process the weights for the given dataset and place them into a new directory.
In order to load these weights on the Pynq, place the resultant folder into the XILINX_BNN_ROOT/data directory on the Pynq device.

## Installing the Training Environment:

There are many ways to set up the training environment, the following steps work on a Ubuntu 16.04 base image on Amazon Web Services (AWS) on a GPU instance.
In order to get an AWS EC2 instance up and running, see the [getting started guide](https://aws.amazon.com/ec2/getting-started/).
If you're using a shared machine, it's strongly suggested you install the python packages (all python and pip commands) under a [virtualenv](https://virtualenv.pypa.io/) environment.

At a high level, the instuctions perform the following steps:

1. Install Nvidia drivers, CUDA and cuDNN
1. Install python packages (Theano, Lasagne, Numpy, Pylearn2)
1. Download datasets

### Install Nvidia Drivers, CUDA and cuDNN

1.  Fetch and install the latest the latest Nvidia CUDA repository package from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
    This can be achieved as follows:

    ```bash
    $ wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    $ sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    $ sudo apt-get update
    ```

    If the 2nd command above hangs, make sure your proxy is set.

1. Install CUDA:

    ```bash
    $ sudo apt-get install cuda -y
    ```

1.  Install cuDNN by download the runtime and development packages from [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn).
    If you have not previously done so, you may need to register an account with Nvidia. Once you've downloaded the packages install them as follows:

    ```bash
    $ sudo dpkg -i libcudnn5_5.1.10-1+cuda8.0_amd64.deb libcudnn5-dev_5.1.10-1+cuda8.0_amd64.deb
    ```

1.  Add CUDA and cuDNN to your library path as follows:

    ```bash
    $ sudo sh -c "echo 'CUDA_HOME=/usr/local/cuda' >> /etc/profile.d/cuda.sh"
    $ sudo sh -c "echo 'export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:\${CUDA_HOME}/lib64' >> /etc/profile.d/cuda.sh"
    $ sudo sh -c "echo 'export LIBRARY_PATH=\${LIBRARY_PATH}:\${CUDA_HOME}/lib64' >> /etc/profile.d/cuda.sh"
    $ sudo sh -c "echo 'export C_INCLUDE_PATH=\${C_INCLUDE_PATH}:\${CUDA_HOME}/include' >> /etc/profile.d/cuda.sh"
    $ sudo sh -c "echo 'export CXX_INCLUDE_PATH=\${CXX_INCLUDE_PATH}:\${CUDA_HOME}/include' >> /etc/profile.d/cuda.sh"
    $ sudo sh -c "echo 'export PATH=\${PATH}:\${CUDA_HOME}/bin' >> /etc/profile.d/cuda.sh"
    ```

1. Reboot your machine, once rebooted check that the Nvidia driver is working by running:

    ```bash
    $ nvidia-smi
    ```

    This command should display some information about your Nvidia GPU and driver and will give an error message if it's not working.

    Note, at the time of writing this, the command at step 2 doesn't appear install the Nvidia drivers properly.
    If it doesn't work, install the previous version of the Nvidia drivers and CUDA as follows:

    ```bash
    $ sudo apt-get autoremove --purge cuda  # If you need to uninstall the other version
    $ sudo apt-get install cuda-drivers=367.48-1 cuda-8-0=8.0.44-1 cuda-runtime-8-0=8.0.44-1 -y
    ```

### Install Python Packages

Remember, if you're using a shared machine it's highly recommended that you install the python packages within a sandboxed python environment, such as [virtualenv](https://virtualenv.pypa.io/).
Alternatively, you can install the packages into your user's site packages location.

1. Install some dependencies:

    ```bash
    $ sudo apt-get install git python-dev libopenblas-dev liblapack-dev gfortran -y
    ```

1. Firstly, install the latest version of pip as follows:

    ```bash
    $ wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py --user
    ```

    If pip is already installed, this command will have no effect.

1. Install Theano, Lasagne:

    ```bash
    $ pip install --user git+https://github.com/Theano/Theano.git@rel-0.9.0beta1
    $ pip install --user https://github.com/Lasagne/Lasagne/archive/master.zip
    ```

    Create a .theanorc configuration file and populate it as follows:

    ```bash
    $ echo "[global]" >> ~/.theanorc
    $ echo "floatX = float32" >> ~/.theanorc
    $ echo "device = gpu" >> ~/.theanorc
    $ echo "openmp = True" >> ~/.theanorc
    $ echo "openmp_elemwise_minsize = 200000" >> ~/.theanorc
    $ echo "" >> ~/.theanorc
    $ echo "[nvcc]" >> ~/.theanorc
    $ echo "fastmath = True" >> ~/.theanorc
    $ echo "" >> ~/.theanorc
    $ echo "[blas]" >> ~/.theanorc
    $ echo "ldflags = -lopenblas" >> ~/.theanorc
    ```

    Create a variable to specify the number of CPU threads you want to limit Theano to.
    If you want limit Theano to the number of threads you have available use the following:

    ```bash
    $ export OMP_NUM_THREADS=`nproc`
    ```

1. Install Pylearn2:

    ```bash
    $ pip install --user numpy==1.11.0 # Pylearn2 seems to not work with the latest version of numpy
    $ git clone https://github.com/lisa-lab/pylearn2
    $ cd pylearn2
    $ python setup.py develop --user
    $ cd ..
    ```

### Download Datasets

1. Download the MNIST and CIFAR10 datasets:

    ```bash
    $ export PYLEARN2_DATA_PATH=~/.pylearn2
    $ mkdir -p ~/.pylearn2
    $ cd pylearn2/pylearn2/scripts/datasets
    $ python download_mnist.py
    $ ./download_cifar10.sh
    $ cd ../../..
    ```

    Occasionally the server with the MNIST dataset goes down, and the download will fail.
    If this happens try running the script at another time.

## Acknowledgements

The source code in this directory was originally forked from [https://github.com/MatthieuCourbariaux/BinaryNet](https://github.com/MatthieuCourbariaux/BinaryNet) and is based on the following publication:
[Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.](http://arxiv.org/abs/1602.02830)

