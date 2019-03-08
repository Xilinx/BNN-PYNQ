FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Uncomment below to have the docker use a proxy.
# Change the URL to refer to your preferred proxy.
#ENV http_proxy http://my.proxy:8080
#ENV https_proxy http://my.proxy:8080
#ENV HTTP_PROXY http://my.proxy:8080
#ENV HTTPS_PROXY http://my.proxy:8080
#RUN echo 'Acquire::http::proxy \"http://my.proxy:8080/\";' >> /etc/apt/apt.conf.d/00proxy
#RUN echo 'Acquire::https::proxy \"http://my.proxy:8080/\";' >> /etc/apt/apt.conf.d/00proxy

# Install some prerequisites
RUN apt-get update &&\
    apt-get upgrade -y &&\
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --fix-missing\
        g++\
        gfortran\
        git\
        patch\
        python-dev\
        python-virtualenv\
        liblapack-dev\
        libopenblas-dev\
        unzip\
        virtualenv\
        wget\
        &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/*

# Set up the python virtualenv: theano, lasagne and pylearn2
RUN virtualenv /root/mlpy
RUN cd /root &&\
    . mlpy/bin/activate &&\
    pip install --upgrade pip &&\
    pip install git+https://github.com/Theano/Theano.git@rel-0.9.0beta1 &&\
    pip install https://github.com/Lasagne/Lasagne/archive/master.zip &&\
    pip install Pillow &&\
    pip install bitstring &&\
    pip install numpy==1.11.0 &&\
    mkdir -p mlpy/src &&\
    cd mlpy/src &&\
    git clone https://github.com/lisa-lab/pylearn2 &&\
    cd pylearn2 &&\
    python setup.py develop &&\
    rm -rf /root/.cache/pip

# Set .bashrc for virtualenv PYLEARN2, THEANO
# Change number of threads to match your environment
RUN echo "source /root/mlpy/bin/activate" >> /root/.bashrc &&\
    echo "export PYLEARN2_DATA_PATH=/root/.pylearn2" >> /root/.bashrc &&\
    echo "export OMP_NUM_THREADS=4" >> /root/.bashrc

# Set the theano configuration file
RUN echo "[global]" >> /root/.theanorc &&\
    echo "floatX = float32" >> /root/.theanorc &&\ 
    echo "device = gpu" >> /root/.theanorc &&\ 
    echo "openmp = True" >> /root/.theanorc &&\ 
    echo "openmp_elemwise_minsize = 200000" >> /root/.theanorc &&\
    echo "" >> /root/.theanorc &&\
    echo "[nvcc]" >> /root/.theanorc &&\
    echo "fastmath = True" >> /root/.theanorc &&\
    echo "" >> /root/.theanorc &&\
    echo "[blas]" >> /root/.theanorc &&\
    echo "ldflags = -lopenblas" >> /root/.theanorc

# Make script to download datasets 
RUN echo '#!/bin/bash' >> /root/download_cifar10_mnist.sh &&\
    echo 'cd /root/mlpy/src/pylearn2/pylearn2/scripts/datasets' >> /root/download_cifar10_mnist.sh &&\
    echo 'python download_mnist.py' >> /root/download_cifar10_mnist.sh &&\
    echo './download_cifar10.sh' >> /root/download_cifar10_mnist.sh &&\
    chmod 755 /root/download_cifar10_mnist.sh

CMD [ "/bin/bash" ]
