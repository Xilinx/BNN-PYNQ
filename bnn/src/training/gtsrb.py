#BSD 3-Clause License
#=======
#
#Copyright (c) 2018, Xilinx Inc.
#All rights reserved.
#
#Based Matthieu Courbariaux's CIFAR-10 example code
#Copyright (c) 2015-2016, Matthieu Courbariaux
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the copyright holder nor the names of its 
#      contributors may be used to endorse or promote products derived from 
#      this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
#EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
#DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

import sys
import os
import time
import math
from argparse import ArgumentParser

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
#import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip
from glob import glob

from PIL import Image

import quantized_net
import cnv
from readTrafficSigns import readTrafficSigns
import augmentors

from collections import OrderedDict

# Pad images with zeros if they are not square.
def square_image(x):
    c = x.shape[0]
    w = x.shape[1]
    h = x.shape[2]
    if w > h:
        fpad = float(w-h)/2
        upad = int(fpad)
        dpad = int(math.ceil(fpad))
        xpu = np.zeros([c, w, upad], dtype=np.float32)
        xpd = np.zeros([c, w, dpad], dtype=np.float32)
        xs = np.append(np.append(xpu, x, axis=2), xpd, axis=2)
    elif w < h:
        fpad = float(h-w)/2
        lpad = int(fpad)
        rpad = int(math.ceil(fpad))
        xpl = np.zeros([c, lpad, h], dtype=np.float32)
        xpr = np.zeros([c, rpad, h], dtype=np.float32)
        xs = np.append(np.append(xpl, x, axis=1), xpr, axis=1)
    else:
        xs = x
    return xs

# Resize images and rescale if requested.
def resize_and_format(image_path, width, normalise=True):

    Xraw, Yraw = readTrafficSigns(image_path)

    X = np.zeros([len(Yraw), Xraw[0].shape[-1], width, width], dtype=np.float32)
    Y = np.zeros(len(Yraw), dtype=np.int32)

    for i, xi, yi in zip(range(len(Yraw)), Xraw, Yraw):
        y = int(yi)
        im = Image.fromarray(xi)
        orig_size = im.size
        ratio = float(width)/max(orig_size)
        new_size = (int(round(ratio*orig_size[0])), int(round(ratio*orig_size[1])))
        xi = np.transpose(np.asarray(im.resize(new_size)), [2,0,1])
        if normalise:
            xi = 2*(xi/255.0) - 1
        x = square_image(xi)
        X[i] = x
        Y[i] = y

    return X, Y

# Get the 'junk class' images, and return the associated label
def get_junk_class(image_path, width, label, normalise=True):

    image_files = glob(image_path + "/*.jpg")
    N = len(image_files)
    X = np.zeros([N, 3, width, width], dtype=np.float32)
    Y = np.zeros(N, dtype=np.int32)
    y = label

    for i, image_file in zip(range(N), image_files):
        im = Image.open(image_file)
        orig_size = im.size
        ratio = float(width)/max(orig_size)
        new_size = (int(round(ratio*orig_size[0])), int(round(ratio*orig_size[1])))
        xi = np.transpose(np.asarray(im.resize(new_size)), [2,0,1])
        if normalise:
            xi = 2*(xi/255.0) - 1
        x = square_image(xi)
        X[i] = x
        Y[i] = y

    return X, Y

if __name__ == "__main__":
    # Parse some command line options
    parser = ArgumentParser(
        description="Train the CNV network on the GTSRB dataset")
    parser.add_argument('-ab', '--activation-bits', type=int, default=1, choices=[1],
        help="Quantized the activations to the specified number of bits, default: %(default)s")
    parser.add_argument('-wb', '--weight-bits', type=int, default=1, choices=[1],
        help="Quantized the weights to the specified number of bits, default: %(default)s")
    parser.add_argument(
        '-f', '--final', action='store_true', default=False,
        help="Use the 'Final' training and test sets (default: %(default)s)")
    parser.add_argument(
        '-ip', '--image-path', type=str, required=True,
        help="Path to the root directory of GTRSB images (e.g., ./GTRSB")
    args = parser.parse_args()

    image_path = args.image_path
    final = args.final

    learning_parameters = OrderedDict()

    # Quantization parameters
    learning_parameters.activation_bits = args.activation_bits
    print("activation_bits = "+str(learning_parameters.activation_bits))
    learning_parameters.weight_bits = args.weight_bits
    print("weight_bits = "+str(learning_parameters.weight_bits))

    # BN parameters
    batch_size = 50
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    learning_parameters.alpha = .1
    print("alpha = "+str(learning_parameters.alpha))
    learning_parameters.epsilon = 1e-4
    print("epsilon = "+str(learning_parameters.epsilon))
    
    # W_LR_scale = 1.    
    learning_parameters.W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(learning_parameters.W_LR_scale))
    
    # Training parameters
    num_epochs = 1000
    print("num_epochs = "+str(num_epochs))
    
    # Decaying LR 
    LR_start = 0.001
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    save_path = "gtsrb-%dw-%da.npz" % (learning_parameters.weight_bits, learning_parameters.activation_bits)
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))

    print('Loading GTSRB dataset...')
    
    image_width = 32
    cropping_percentage = 0.25
    oversize_pixels = int(image_width * cropping_percentage)
    if not final:
        Xin, Yin = resize_and_format(image_path + "/Training", image_width + oversize_pixels)
    else:
        Xin, Yin = resize_and_format(image_path + "/Final_Training", image_width + oversize_pixels)

    print(Xin.shape)
    print(Yin.shape)
    # Each sign has 30 duplicates of the same sign, but from different distances.
    # The models generalize much better if split the dataset up based on each sign,
    # rather than each image.
    num_dupe = 30
    num_diff = len(Yin) / num_dupe
    Xin = Xin.reshape([num_diff,num_dupe,Xin.shape[1],Xin.shape[2],Xin.shape[3]])
    Yin = Yin.reshape([num_diff,num_dupe])
    print(Xin.shape)
    print(Yin.shape)

    shuffle = True
    if shuffle:
        rind = np.array(range(len(Yin)))
        np.random.shuffle(rind) # In place shuffle of indices.
        Xin = Xin[rind,:,:,:,:]
        Yin = Yin[rind,:]

    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    num_examples = len(Yin)
    num_outputs = len(np.unique(Yin))
    print("Number of classes = %i" % (num_outputs))
    val_split = int(num_examples*0.8)
    test_split = int(num_examples*0.9)
    train_set = OrderedDict()
    valid_set = OrderedDict()
    test_set = OrderedDict()
    train_set.X = np.copy(Xin[:val_split,:,:,:,:])
    valid_set.X = np.copy(Xin[val_split:test_split,:,:,:,:])
    test_set.X = np.copy(Xin[test_split:,:,:,:,:])
    
    # flatten targets
    train_set.y = Yin[:val_split,:]
    valid_set.y = Yin[val_split:,:]
    test_set.y = Yin[test_split:,:]
    print("Train shape: "+str(train_set.X.shape))
    print("Valid shape: "+str(valid_set.X.shape))
    print("Test shape: "+str(test_set.X.shape))
    train_set.X = train_set.X.reshape([train_set.X.shape[0]*train_set.X.shape[1],train_set.X.shape[2],train_set.X.shape[3],train_set.X.shape[4]])
    valid_set.X = valid_set.X.reshape([valid_set.X.shape[0]*valid_set.X.shape[1],valid_set.X.shape[2],valid_set.X.shape[3],valid_set.X.shape[4]])
    test_set.X = test_set.X.reshape([test_set.X.shape[0]*test_set.X.shape[1],test_set.X.shape[2],test_set.X.shape[3],test_set.X.shape[4]])
    train_set.y = train_set.y.reshape([train_set.y.shape[0]*train_set.y.shape[1]])
    valid_set.y = valid_set.y.reshape([valid_set.y.shape[0]*valid_set.y.shape[1]])
    test_set.y = test_set.y.reshape([test_set.y.shape[0]*test_set.y.shape[1]])
    junk_class = True
    if junk_class:
        Xj, Yj = get_junk_class(image_path + "/junk", image_width + oversize_pixels, num_outputs)
        num_junk = len(Yj)
        print("Adding an extra %i examples to \"junk\" class." % (num_junk))
        num_outputs += 1
        junk_val_split = int(num_junk*0.8)
        junk_test_split = int(num_junk*0.9)
        train_set.X = np.append(train_set.X, Xj[:junk_val_split], axis=0)
        valid_set.X = np.append(valid_set.X, Xj[junk_val_split:junk_test_split], axis=0)
        test_set.X = np.append(test_set.X, Xj[junk_test_split:], axis=0)
        train_set.y = np.append(train_set.y, Yj[:junk_val_split], axis=0)
        valid_set.y = np.append(valid_set.y, Yj[junk_val_split:junk_test_split], axis=0)
        test_set.y = np.append(test_set.y, Yj[junk_test_split:], axis=0)

    if shuffle:
        print("Shuffling training set.")
        rind = np.array(range(len(train_set.y)))
        np.random.shuffle(rind) # In place shuffle of indices.
        train_set.X = train_set.X[rind,:,:,:]
        train_set.y = train_set.y[rind]
    print("Train shape: "+str(train_set.X.shape))
    print("Valid shape: "+str(valid_set.X.shape))
    print("Test shape: "+str(test_set.X.shape))

    print("Applying rotations and cropping.")
    rotate_train = False
    rotate_valid = True
    rotate_test = True
    random_rot_range = [-7, 7]
    random_rot_factor = 1
    if rotate_train:
        (train_set.X, train_set.y) = augmentors.random_rotations(train_set.X, train_set.y, random_rot_range, random_rot_factor, extend=False)
    if rotate_valid:
        (valid_set.X, valid_set.y) = augmentors.random_rotations(valid_set.X, valid_set.y, random_rot_range, random_rot_factor, extend=False)
    if rotate_test:
        (test_set.X, test_set.y) = augmentors.random_rotations(test_set.X, test_set.y, random_rot_range, random_rot_factor, extend=False)
    crop_train = False
    crop_valid = True
    crop_test = True
    random_crop_range = (oversize_pixels, oversize_pixels)
    random_crop_factor = 1
    cropped_size = (image_width,image_width)
    if crop_train:
        (train_set.X, train_set.y) = augmentors.random_crop(train_set.X, train_set.y, random_crop_range, random_crop_factor, cropped_size, extend=False)
    if crop_valid:
        (valid_set.X, valid_set.y) = augmentors.random_crop(valid_set.X, valid_set.y, random_crop_range, random_crop_factor, cropped_size, extend=False)
    if crop_test:
        (test_set.X, test_set.y) = augmentors.random_crop(test_set.X, test_set.y, random_crop_range, random_crop_factor, cropped_size, extend=False)

    print("Train shape: "+str(train_set.X.shape))
    print("Valid shape: "+str(valid_set.X.shape))
    print("Test shape: "+str(test_set.X.shape))
    
    # Onehot the targets
    train_set.y = np.float32(np.eye(num_outputs)[train_set.y])    
    valid_set.y = np.float32(np.eye(num_outputs)[valid_set.y])
    test_set.y = np.float32(np.eye(num_outputs)[test_set.y])
    
    # for hinge loss
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.
    
    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = cnv.genCnv(input, num_outputs, learning_parameters)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    # W updates
    W = lasagne.layers.get_all_params(cnn, quantized=True)
    W_grads = quantized_net.compute_grads(loss,cnn)
    updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
    updates = quantized_net.clipping_scaling(updates,cnn)
    
    # other parameters updates
    params = lasagne.layers.get_all_params(cnn, trainable=True, quantized=False)
    updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    quantized_net.train(
            train_fn,val_fn,
            cnn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set.X,train_set.y,
            valid_set.X,valid_set.y,
            test_set.X,test_set.y,
            save_path=save_path,
            shuffle_parts=shuffle_parts,
            rotations=random_rot_range)
