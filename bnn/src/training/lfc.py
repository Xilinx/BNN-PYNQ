#BSD 3-Clause License
#=======
#
#Copyright (c) 2017, Xilinx Inc.
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

import lasagne
import quantization as q
import quantized_net as qn

def genLfc(input, num_outputs, learning_parameters):
    # A function to generate the lfc network topology which matches the overlay for the Pynq board.
    # WARNING: If you change this file, it's likely the resultant weights will not fit on the Pynq overlay.
    if num_outputs < 1 or num_outputs > 64:
        error("num_outputs should be in the range of 1 to 64.")
    num_units = 1024
    n_hidden_layers = 3
    if learning_parameters.activation_bits == 1:
        act_quant = q.QuantizationBinary()
    else:
        act_quant = q.QuantizationFixed(learning_parameters.activation_bits,
            learning_parameters.activation_bits - 2)
    activation = qn.FixedHardTanH(act_quant)
    if learning_parameters.weight_bits == 1:
        weight_quant = q.QuantizationBinary()
    else:
        weight_quant = q.QuantizationFixed(learning_parameters.weight_bits,
            learning_parameters.weight_bits - 2)
    W_LR_scale = learning_parameters.W_LR_scale
    epsilon = learning_parameters.epsilon
    alpha = learning_parameters.alpha
    dropout_in = learning_parameters.dropout_in
    dropout_hidden = learning_parameters.dropout_hidden

    mlp = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input)
            
    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=dropout_in)
    
    for k in range(n_hidden_layers):

        mlp = qn.DenseLayer(
                mlp, 
                quantization=weight_quant,
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
    
    mlp = qn.DenseLayer(
                mlp, 
                quantization=weight_quant,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_outputs)
    
    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha)
    return mlp

