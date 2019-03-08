# BNN-PYNQ Dockerfiles

These directories contain Dockerfiles for building docker images for setting up the training environment with or without access to GPUs.

## Prerequisites

These images require a working installations of the following:
* [docker](https://www.docker.com/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) if you wish to train using a GPU (recommended)

## Building the Docker Images

The docker images can be built as follows:

```bash
docker build -t bnn-pynq:<cpu|gpu> <cpu|gpu>
```

Note, the user must have sufficient privileges to use docker.

## Running the Image

Once you have a built docker image, you can run the image and begin training.
You can instantiate the docker image as follows:

```bash
docker run -i -t -v /path/to/BNN-PYNQ:/root/BNN-PYNQ -v /dataset/directory:/root/.pylearn2 bnn-pynq:<cpu|gpu> /bin/bash
```

Note, `/path/to/BNN-PYNQ` refers to the path on the host where you have cloned the BNN-PYNQ repository and
`/dataset/directory` refers to the location on the host where you would like to download the Cifar10 and MNIST datasets.

## Training the Networks

Once you've instantiated the docker image and have gained access to a shell (see [running the image](#running-the-Image)), we've provided a convenient script to download the Cifar10 and MNIST datasets.
Run the following to download the datasets:

```bash
/root/download_cifar10_mnist.sh
```

Now you can `cd` to the directory containing the BNN-PYNQ training source code:
```bash
cd /root/BNN-PYNQ/bnn/src/training/
```

From here, you can follow the instructions [here](../) to train the networks.

## Troubleshooting

### Using a Proxy

If you need to use the docker images behind a proxy,
please uncomment and edit the lines in the desired `Dockerfile` to match your proxy settings.

