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

import theano
import theano.tensor as T
import numpy as np
from functools import partial

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

# Our own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz, 
        
round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

# From https://github.com/MatthieuCourbariaux/BinaryNet
def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1 
# during back propagation
# From https://github.com/MatthieuCourbariaux/BinaryNet
def binary_tanh_unit(x):
    return 2.*round3(hard_sigmoid(x))-1.

# From https://github.com/MatthieuCourbariaux/BinaryNet
def binary_sigmoid_unit(x):
    return round3(hard_sigmoid(x))

# A floating point quantization scheme (to easily check how non-quantized layers would have worked)
class QuantizationFloat(object):
    def __init__(self, min=-3.40e38, max=3.4e38):
        self.min = min
        self.min = max

    # Quantize activations
    def quantize(self, X):
        return T.clip(X, self.min, self.max)

    # Quantize weights (can use Theano's built-in round function - should be faster)
    def quantizeWeights(self, X):
        return T.clip(X, self.min, self.max)

    def clipWeights(self, X):
        return T.clip(X, self.min, self.max)

# A binary quantization scheme, based on this example:
# https://github.com/MatthieuCourbariaux/BinaryNet
class QuantizationBinary(object):
    def __init__(self, scale=1.0):
        self.scale = scale
        self.min = -scale
        self.max = scale

    # Quantize activations
    def quantize(self, X):
        return binary_tanh_unit(X / self.scale)*self.scale

    # Quantize weights (can use Theano's built-in round function - should be faster?)
    def quantizeWeights(self, X):
        # [-1,1] -> [0,1]
        Xa = hard_sigmoid(X / self.scale)
        Xb = T.round(Xa)
        # 0 or 1 -> -1 or 1
        return T.cast(T.switch(Xb,self.scale,-self.scale), theano.config.floatX)

    def clipWeights(self, X):
        return T.clip(X, -self.scale, self.scale)

# A fixed point quantization scheme
class QuantizationFixed(object):
    def __init__(self, wordlength, fraclength, narrow_range=True):
        self.wordlength = wordlength
        self.fraclength = fraclength
        self.narrow_range = narrow_range
        self.set_quantization_params()

    # Set up the parameters for run-time quantization
    def set_quantization_params(self):
        self.set_min_max()
        self.set_scale_shift()

    # Work out how much to shift and scale prior to rounding in the current scheme
    def set_scale_shift(self):
        if self.narrow_range:
            sub = 2
        else:
            sub = 1
        self.scale = (2.**self.wordlength - sub) / (self.max - self.min)
        self.shift = -self.min

    # Find the minimum and maximum representable parameters in this scheme
    def set_min_max(self):
        min_val = - (2.**(self.wordlength - self.fraclength - 1))
        max_val = - min_val - 2.**-self.fraclength
        if self.narrow_range:
            min_val = - max_val
        self.min = min_val
        self.max = max_val

    # Quantize activations
    def quantize(self, X):
        return T.cast(round3((T.clip(X, self.min, self.max) + self.shift)*self.scale) / self.scale - self.shift, theano.config.floatX)

    # Quantize weights (can use Theano's built-in round function - should be faster)
    def quantizeWeights(self, X):
        return T.cast(T.round((T.clip(X, self.min, self.max) + self.shift)*self.scale) / self.scale - self.shift, theano.config.floatX)

    def clipWeights(self, X):
        return T.clip(X, self.min, self.max)

