#BSD 3-Clause License
#=======
#
#Copyright (c) 2019, Xilinx Inc.
#All rights reserved.
#
#Based on Matthieu Courbariaux's BinaryNet example
#Copyright (c) 2015-2016, Matthieu Courbariaux
#All rights reserved
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

import time

from collections import OrderedDict

import numpy as np

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import augmentors

# a class to implemented an arbitrarily quantized version of the hard_tanh function.
class FixedHardTanH(object):
    def __init__(self, quantization):
        self.quantization = quantization

    def __call__(self, x):
        y = T.clip(x, -1, 1) 
        return self.quantization.quantize(y)

# This class extends the Lasagne DenseLayer to support quantization.
class DenseLayer(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, 
        quantization, W_LR_scale="Glorot", **kwargs):
        
        self.quantization = quantization
        
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
        super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((self.quantization.min,self.quantization.max)), **kwargs)
        # add the quantized tag to weights            
        self.params[self.W]=set(['quantized']) # I have no idea what this does!

    def quantize(self, X):
        return self.quantization.quantizeWeights(X)

    def clip(self, X):
        return self.quantization.clipWeights(X)

    def get_output_for(self, input, **kwargs):
        self.Wq = self.quantize(self.W)
        Wr = self.W
        self.W = self.Wq
            
        rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
        
        self.W = Wr
        
        return rvalue

# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    
    def __init__(self, incoming, num_filters, filter_size,
        quantization, W_LR_scale="Glorot", **kwargs):
        
        self.quantization = quantization

        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))
            
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
            
        super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((self.quantization.min,self.quantization.max)), **kwargs)

        self.params[self.W]=set(['quantized'])

    def quantize(self, X):
        return self.quantization.quantizeWeights(X)

    def clip(self, X):
        return self.quantization.clipWeights(X)

    def convolve(self, input, **kwargs):
        
        self.Wq = self.quantize(self.W)
        Wr = self.W
        self.W = self.Wq
            
        rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)
        
        self.W = Wr
        
        return rvalue

# This function computes the gradient of the quantized weights
def compute_grads(loss,network):
        
    layers = lasagne.layers.get_all_layers(network)
    grads = []
    
    for layer in layers:
    
        params = layer.get_params(quantized=True)
        if params:
            # print(params[0].name)
            grads.append(theano.grad(loss, wrt=layer.Wq))
                
    return grads

# This functions clips the weights after the parameter update
def clipping_scaling(updates,network):
    
    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(quantized=True)
        for param in params:
            print("W_LR_scale = "+str(layer.W_LR_scale))
            print("H = "+str(layer.quantization.max))
            updates[param] = param + layer.W_LR_scale*(updates[param] - param)
            updates[param] = layer.clip(updates[param])

    return updates
        
# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn,val_fn,
            model,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test,
            save_path=None,
            shuffle_parts=1,
            starting_epoch=1,
            best_val_err=100,
            best_epoch=1,
            test_err=100,
            test_loss=100,
            rotations=None):
    
    # A function which shuffles a dataset
    def shuffle(X,y):
        
        # print(len(X))
        
        chunk_size = len(X)/shuffle_parts
        shuffled_range = range(chunk_size)
        
        X_buffer = np.copy(X[0:chunk_size])
        y_buffer = np.copy(y[0:chunk_size])
        
        for k in range(shuffle_parts):
            
            np.random.shuffle(shuffled_range)

            for i in range(chunk_size):
                
                X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
                y_buffer[i] = y[k*chunk_size+shuffled_range[i]]
            
            X[k*chunk_size:(k+1)*chunk_size] = X_buffer
            y[k*chunk_size:(k+1)*chunk_size] = y_buffer
        
        return X,y
        
        # shuffled_range = range(len(X))
        # np.random.shuffle(shuffled_range)
        
        # new_X = np.copy(X)
        # new_y = np.copy(y)
        
        # for i in range(len(X)):
            
            # new_X[i] = X[shuffled_range[i]]
            # new_y[i] = y[shuffled_range[i]]
            
        # return new_X,new_y
    
    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,y,LR,rotations,oversize_pixels):
        
        loss = 0
        batches = len(X)/batch_size

        if rotations != None:
            (X, y) = augmentors.random_rotations(X, y, rotations, 1, extend=False)
        random_crop = oversize_pixels[0] != 0 or oversize_pixels[1] != 0
        if random_crop:
            cropped_size = (X.shape[2]-oversize_pixels[0], X.shape[3]-oversize_pixels[1])
            (X, y) = augmentors.random_crop(X, y, oversize_pixels, 1, cropped_size, extend=False)
        
        for i in range(batches):
            loss += train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR)
        
        loss/=batches
        
        return loss
    
    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y):
        
        err = 0
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss
        
        err = err / batches * 100
        loss /= batches

        return err, loss
    
    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    starting_epoch -= 1 # epochs are printed as 1 to num_epochs.
    if hasattr(num_epochs, '__int__'):
        epochs = range(starting_epoch, num_epochs)
        LRs = map(lambda e: LR_start*(LR_decay**e), epochs)
    else:
        if len(num_epochs) != len(LR_start):
            print("num_epochs and learning_rate must be the same length")
            exit()
        epochs = range(starting_epoch, num_epochs[-1])
        prev_epoch = 1
        LRs = []
        for lr,e in zip(LR_start, num_epochs):
            LRs += [lr]*(e-prev_epoch)
            prev_epoch = e
        LRs += LR_start[-1:]
        LRs[:starting_epoch] = []
        num_epochs = num_epochs[-1]
    
    oversize_pixels = (X_train.shape[2]-X_val.shape[2], X_train.shape[3]-X_val.shape[3])

    # We iterate over epochs:
    for epoch, LR in zip(epochs, LRs):
        
        start_time = time.time()
        
        train_loss = train_epoch(X_train,y_train,LR,rotations,oversize_pixels)
        X_train,y_train = shuffle(X_train,y_train)
        
        val_err, val_loss = val_epoch(X_val,y_val)
        
        # test if validation error went down
        if val_err <= best_val_err:
            
            best_val_err = val_err
            best_epoch = epoch+1
            
            test_err, test_loss = val_epoch(X_test,y_test)
            
            if save_path is not None:
                np.savez(save_path, *lasagne.layers.get_all_param_values(model))
        
        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
        
        # decay the LR
        LR *= LR_decay
