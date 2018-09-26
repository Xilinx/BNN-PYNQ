
'''
Modified version of the mnist.py training script
Designed to be used with the NIST SD 19 Handprinted Forms and Characters dataset: https://www.nist.gov/srd/nist-special-database-19
Best results have been achieved by cropping all images by bounding box of the character, and scaling to 28x28 to emulate the MNIST dataset.
This training script assumes 
For related work released after this project, see https://www.nist.gov/itl/iad/image-group/emnist-dataset
'''

from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip
import glob
from scipy import misc

import binary_net

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

from collections import OrderedDict
#import matplotlib.pyplot as plt

def loaddata():
    '''
    Loads the NIST SD19 Character dataset, which must can be downloaded from https://www.nist.gov/srd/nist-special-database-19
    Assumes dataset is downloaded in the current directory (..../bnn/src/training) and ordered by class.
    '''

    classes = ["30", "31", "32", "33", "34", "35", "36", "37", "38", "39", #Digits
"41", "42", "43", "44", "45", "46", "47", "48", "49", "4a", "4b", "4c", "4d", "4e", "4f", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "5a", #Upper case
"61", "62", "63", "64", "65", "66", "67", "68", "69", "6a", "6b", "6c", "6d", "6e", "6f", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "7a"] #Lower case

    NumImagesPerClassTrain = 300
    NumImagesPerClassTest = 100
    NumImagesPerClassValidation = 50

    pngTrain = []
    pngTest = []
    pngValidation = []
    labelsTrain = []
    labelsTest = []
    labelsValidation = []

    for glyph in classes:
        i = 0
        print("Loading Glyph code: "+glyph)
        for image_path in glob.glob("./by_class/"+glyph+"/train_"+glyph+"/*.png"):
            if (i < NumImagesPerClassTrain):
                pngTrain.append(misc.imread(image_path)) 
                labelsTrain.append(classes.index(glyph))
                i=i+1
            elif(i < (NumImagesPerClassTrain + NumImagesPerClassValidation)):
                pngValidation.append(misc.imread(image_path)) 
                labelsValidation.append(classes.index(glyph))
                i=i+1
            else:
                break
        k = 0
        for image_path in glob.glob("./by_class/"+glyph+"/hsf_4/*.png"):
            if (k < NumImagesPerClassTest):
                pngTest.append(misc.imread(image_path)) 
                labelsTest.append(classes.index(glyph))
                k=k+1
            else:
                break

    labelsTrain = np.asarray(labelsTrain)
    labelsTrain = np.float32(np.eye(62)[labelsTrain])
    labelsTrain = 2*labelsTrain -1.

    imgTrain = np.asarray(pngTrain)
    imgTrain = np.float32(imgTrain)
    imgTrain = imgTrain[:,:,:,1]
    imgTrain = 1. - 2*(imgTrain[:,np.newaxis,:,:]/255.)#Normalize between -1 and 1 #need to split this operation up

    labelsTest = np.asarray(labelsTest)
    labelsTest = np.float32(np.eye(62)[labelsTest])
    labelsTest = 2*labelsTest -1.

    imgTest = np.asarray(pngTest)
    imgTest = np.float32(imgTest)
    imgTest = imgTest[:,:,:,1]
    imgTest = 1. - 2*(imgTest[:,np.newaxis,:,:]/255.)#Normalize. #Need to split up operations

    labelsValidation = np.asarray(labelsValidation)
    labelsValidation = np.float32(np.eye(62)[labelsValidation])
    labelsValidation = 2*labelsValidation - 1. #Normalize

    imgValidation = np.asarray(pngValidation)
    imgValidation = np.float32(imgValidation)
    imgValidation = imgValidation[:,:,:,1]
    imgValidation = 1. - 2*(imgValidation[:,np.newaxis,:,:]/255.)

    return (imgTrain, labelsTrain, imgTest, labelsTest, imgValidation, labelsValidation)

if __name__ == "__main__":
    
    # BN parameters
    input_size = 128 #Standard NIST SD19 dataset is 128x128 pixels
    batch_size = 100
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # MLP parameters
    num_units = 1024
    print("num_units = "+str(num_units))
    n_hidden_layers = 3
    print("n_hidden_layers = "+str(n_hidden_layers))
    
    # Training parameters
    num_epochs = 1000
    print("num_epochs = "+str(num_epochs))
    
    # Dropout parameters
    dropout_in = .2 # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = .5
    print("dropout_hidden = "+str(dropout_hidden))
    
    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
    
    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Decaying LR 
    LR_start = .003
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    save_path = "char_parameters.npz"
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading Character dataset...')

    train_setX, train_setY, test_setX, test_setY, valid_setX, valid_setY = loaddata()
    
    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = lasagne.layers.InputLayer(
            shape=(None, 1, input_size, input_size),
            input_var=input)
            
    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=dropout_in)
    
    for k in range(n_hidden_layers):

        mlp = binary_net.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)                  
        
        mlp = lasagne.layers.BatchNormLayer(
                mlp,
                epsilon=epsilon, 
                alpha=alpha)

        mlp = lasagne.layers.NonlinearityLayer(
                mlp,
                nonlinearity=activation)
                
        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=dropout_hidden)
    
    mlp = binary_net.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=62)    
    
    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_net.compute_grads(loss,mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,mlp)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    binary_net.train(
            train_fn,val_fn,
            mlp,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_setX,train_setY,
            valid_setX,valid_setY,
            test_setX,test_setY,
            save_path,
            shuffle_parts)