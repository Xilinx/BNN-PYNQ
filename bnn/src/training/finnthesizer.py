# BSD 3-Clause License
# =======

# Copyright (c) 2020, Xilinx
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__authors__ = "Ussama Zahid, Giulio Gambardella, Yaman Umuroglu, Nicholas Fraser, Christoph Doehring"
__copyright__ = "Copyright 2020, Xilinx"

import os
import numpy as np
from functools import reduce

# convenience function to prepare a fully-connected BNN for FINN
# with given number of (SIMD, PE) per layer
# and arbitrary precisions for both weights and activations
def convertFCNetwork(npzFile, targetDirBin, targetDirHLS, simdCounts, peCounts, \
    WeightsPrecisions_fractional, ActivationPrecisions_fractional, InputPrecisions_fractional, \
    WeightsPrecisions_integer, ActivationPrecisions_integer, InputPrecisions_integer):
    
    numLayers = len(simdCounts)
    if not os.path.exists(targetDirBin):
        os.mkdir(targetDirBin)
    if not os.path.exists(targetDirHLS):
        os.mkdir(targetDirHLS)
    # instantiate the weight reader, note how interleaveChannels=False for a 
    # fully connected network
    r = BNNWeightReader(npzFile, False)
    config = "/**\n"
    config+= " * Finnthesizer Config-File Generation\n";
    config+= " *\n **/\n\n"
    config+= "#ifndef __LAYER_CONFIG_H_\n#define __LAYER_CONFIG_H_\n\n"
    for l in range(numLayers):
        print("process layer" + str(l))
        simdCount = simdCounts[l]
        peCount = peCounts[l]
        WPrecision_fractional = WeightsPrecisions_fractional[l]
        APrecision_fractional = ActivationPrecisions_fractional[l]
        IPrecision_fractional = InputPrecisions_fractional[l]
        WPrecision_integer = WeightsPrecisions_integer[l]
        APrecision_integer = ActivationPrecisions_integer[l]
        IPrecision_integer = InputPrecisions_integer[l]
        # read out weights and thresholds
        (w,t) = r.readFCBNComplex(WPrecision_fractional, APrecision_fractional, IPrecision_fractional, \
            WPrecision_integer, APrecision_integer, IPrecision_integer)
        # compute the padded width and height
        paddedH = padTo(w.shape[0], peCount)
        paddedW = padTo(w.shape[1], simdCount)
        if (l==0): # for the first layer, we pad to multiple of 64 due to the AXI interface
            paddedW = padTo(w.shape[1], max(simdCount,64))
        if (l==numLayers-1): # for the last layer, we pad to multiple of 64 due to the AXI interface
            paddedH = padTo(w.shape[0], max(peCount,64))
        # compute memory needed for weights and thresholds
        neededWMem = (paddedW * paddedH) // (simdCount * peCount)
        neededTMem = paddedH // peCount
        print("Layer %d: %d x %d, SIMD = %d, PE = %d" % (l, paddedH, paddedW, simdCount, peCount))
        print("WMem = %d TMem = %d" % (neededWMem, neededTMem))
        print("IPrecision = %d.%d WPrecision = %d.%d APrecision = %d.%d" % (IPrecision_integer, IPrecision_fractional, \
            WPrecision_integer,WPrecision_fractional, APrecision_integer, APrecision_fractional))
        # instantiate PE memory generator
        m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem, WPrecision_integer, APrecision_integer, IPrecision_integer, \
            WPrecision_fractional, APrecision_fractional, IPrecision_fractional)  
        #add layer to config
        config += (printFCDefines("L%d" % l, simdCount, peCount, neededWMem, neededTMem, paddedW, paddedH, \
            WPrecision_integer, APrecision_integer, WPrecision_fractional, APrecision_fractional)) + "\n" 
        
        # pack parameters into PE memory
        m.addMatrix(w,t,paddedW,paddedH)
        
        # create HLS weight init files for initializing memory contents directly
        # while generating the bitstream
        # m.createHLSInitFiles(targetDirHLS + "/memdata-" + str(l) + ".h", str(l))
        
        # create binary weight files -- useful for runtime initialization since
        # HLS might freeze / not work for very large header files
        # note that it will still be necessary to declare the PE memories in 
        m.createBinFiles(targetDirBin, str(l))
    
    # create parameter files for tiny-cnn
    config+="#endif //__LAYER_CONFIG_H_\n"
    configFile = open(targetDirHLS+"/config.h", "w")
    configFile.write(config)
    configFile.close()

# return HW config string as C #define's for a Conv layer
def printConvDefines(prefix, kernelDim, ifm_ch, ifm_dim, ofm_ch, ofm_dim, simd, pe, wmem, tmem, wpi, api, wpf, apf):
    #network topology
    config = ""
    numb_ops = 2*ifm_ch*ofm_ch*kernelDim*kernelDim*ofm_dim*ofm_dim # 2* because of MAC
    est_latency = numb_ops/(2*simd*pe)
    config += "/**\n * Convolutional Layer %s:\n *      IFM  = %5d  IFM_CH = %5d\n *      OFM  = %5d  OFM_CH = %5d\n *     SIMD  = %5d    PE   = %5d\n *     WMEM  = %5d   TMEM  = %5d\n *     #Ops  = %5d   Ext Latency  = %5d\n**/\n" % (prefix, ifm_dim, ifm_ch, ofm_dim, ofm_ch, simd, pe, wmem, tmem, numb_ops, est_latency)       
    config += "\n" + "#define %s_K %d"       % (prefix, kernelDim)
    config += "\n" + "#define %s_IFM_CH %d"  % (prefix, ifm_ch)
    config += "\n" + "#define %s_IFM_DIM %d" % (prefix, ifm_dim)
    config += "\n" + "#define %s_OFM_CH %d"  % (prefix, ofm_ch) 
    config += "\n" + "#define %s_OFM_DIM %d" % (prefix, ofm_dim) 
    #network configuration
    config += "\n" + "#define %s_SIMD %d"  % (prefix, simd)
    config += "\n" + "#define %s_PE %d"    % (prefix, pe)
    config += "\n" + "#define %s_WMEM %d"  % (prefix, wmem)
    config += "\n" + "#define %s_TMEM %d"  % (prefix, tmem) 
    #precision used
    config += "\n" + "#define %s_WPI %d"   % (prefix, wpi)
    config += "\n" + "#define %s_API %d"   % (prefix, api)
    config += "\n" + "#define %s_WPF %d"   % (prefix, wpf)
    config += "\n" + "#define %s_APF %d\n" % (prefix, apf)
    return config

# return HW config string as C #define's for a FC layer
def printFCDefines(prefix, simd, pe, wmem, tmem, mw, mh, wpi, api, wpf, apf):
    config = ""
    numb_ops = 2*mw*mh # 2* because of MAC
    est_latency = numb_ops/(2*simd*pe)
    config += "/**\n * Fully-Connected Layer %s:\n *     MatW = %5d MatH = %5d\n *     SIMD = %5d  PE  = %5d\n *     WMEM = %5d TMEM = %5d\n *     #Ops  = %5d   Ext Latency  = %5d\n**/\n" % (prefix, mw, mh, simd, pe, wmem, tmem, numb_ops, est_latency)       
    config += "\n" + "#define %s_SIMD %d" % (prefix, simd)
    config += "\n" + "#define %s_PE %d" % (prefix, pe)
    config += "\n" + "#define %s_WMEM %d" % (prefix, wmem)
    config += "\n" + "#define %s_TMEM %d" % (prefix, tmem)
    config += "\n" + "#define %s_MW %d" % (prefix, mw)
    config += "\n" + "#define %s_MH %d" % (prefix, mh)
    config += "\n" + "#define %s_WPI %d" % (prefix, wpi)
    config += "\n" + "#define %s_API %d" % (prefix, api)
    config += "\n" + "#define %s_WPF %d" % (prefix, wpf)
    config += "\n" + "#define %s_APF %d\n" % (prefix, apf)
    return config

# return val to nearest multiple of pad
def padTo(val, pad):
    rem = val % pad
    return val if rem == 0 else (val + pad - rem)

# the quantization function
def quantize(x, integer, fract):
    bits=integer+fract
    if (bits==1):
        return(binarize(x))
    n = float(2**fract) # GIULIO ADD CLIP
    return np.floor(x * n + 0.5) / n    

# the binarization function, basically the sign encoded as 1 for positive and
# 0 for negative
def binarize(w):
    return np.where(w < 0, 0, 1)

# convert a fully connected layer plus batch normalization into 
# the simplified form (quantized weight and multiple thresholds)
# note that the neurons are assumed to be in the columns of the weight matrix
def makeFCBNComplex(weights, bias, beta, gamma, mean, invstd, WPrecisions_fract, APrecisions_fract, \
    WPrecisions_int, APrecisions_int, usePopCount=False, use_rowmajor=True, numThresBits=16, numThresIntBits=None):
    
    ins = weights.shape[0]
    outs = weights.shape[1]
    APrecision = APrecisions_fract + APrecisions_int
    print("Extracting FCBN complex, ins = %d outs = %d" % (ins, outs))
    # compute a preliminary thresholds from the batchnorm parameters
    if (APrecision == 1):
        step = np.zeros(1, dtype=np.float64)
    else:
        # This one make -0.5 and +0.5 with 2 bits
        step = np.linspace(-1,1,num=2**(APrecision-1), endpoint=False, dtype=np.float64) + 1./(2**(APrecisions_fract+1))
        # step = np.linspace(-1,1,num=2**APrecision-1,endpoint=False) # Equidistant points between -1 and +1 (hardtanh)
        # step = step[1:] # Removing the -1 point for symmetrical quantization - hardtanh
    thresholds = np.zeros((len(step),len(mean)), dtype=np.float64)
    for i in range(len(step)):
        thresholds[i] = (mean - bias) + ((step[i] - beta) / (gamma*invstd))
    # ensure all neurons activate on the "positive" side, so we can use
    # greater-than-threshold activation
    need_flip = np.sign(gamma)
    factor = need_flip if numThresIntBits is None else need_flip * 2**(numThresBits - numThresIntBits)
    thresholds = factor*thresholds
    # turn threshold into "number of 1s" (popcount) instead of signed sum
    if usePopCount:
        thresholds = (ins + thresholds)/2
    # Integer-like threshold
    else:
        thresholds = np.where(need_flip < 0, np.ceil(thresholds), np.floor(thresholds))
    thresholds = thresholds.transpose(1,0).astype(np.int)

    # generating weights
    weights = weights * need_flip
    weights = quantize(weights, WPrecisions_int, WPrecisions_fract)
    # note how we change from col major to row major if requested
    if use_rowmajor:
        weights = weights.transpose(1,0)
    return (weights, thresholds)

# binarize and pack convolutional layer weights into a matrix and compute
# thresholds from the conv bias and batchnorm parameters
def makeConvBNComplex(weights, bias, beta, gamma, mean, invstd, interleaveChannels, \
    WPrecisions_fract, APrecisions_fract, IPrecisions_fract, \
    WPrecisions_int, APrecisions_int, IPrecisions_int, usePopCount=True, numThresBits=16, numThresIntBits=None):
    
    WPrecision = WPrecisions_fract + WPrecisions_int
    APrecision = APrecisions_fract + APrecisions_int
    IPrecision = IPrecisions_fract + IPrecisions_int
    numOut = weights.shape[0]
    numIn = weights.shape[1]
    k = weights.shape[2]
    # the fanin is used to ensure positive-only threshold
    fanin = numIn * k * k
    if(k != weights.shape[3]):
        raise Exception("Nonsymmetric conv kernels are not yet supported")
    print("Extracting conv-BN complex, OFM=%d IFM=%d k=%d" % (numOut, numIn, k))

    # compute a preliminary threshold from the batchnorm parameters,
    # subtracting the conv bias from the batchnorm mean
    if (APrecision == 1):
        step = np.zeros(1, dtype=np.float64)
    else:
        # This one make -0.5 and +0.5 with 2 bits
        step = np.linspace(-1,1,num=2**(APrecision-1), endpoint=False, dtype=np.float64) + 1./(2**(APrecisions_fract+1)) 
        # step = np.linspace(-1,1,num=2**APrecision-1,endpoint=False) # Equidistant points between -1 and +1 (hardtanh)
        # step = step[1:] # Removing the -1 point for symmetrical quantization - hardtanh
    thresholds = np.zeros((len(step),len(mean)), dtype=np.float64)
    for i in range(len(step)):
        thresholds[i] = (mean - bias) + ((step[i] - beta) / (gamma*invstd))
    # ensure all neurons activate on the "positive" side, so we can use
    # greater-than-threshold activation
    need_flip = np.sign(gamma)
    factor = need_flip if numThresIntBits is None else need_flip * 2**(numThresBits - numThresIntBits)
    thresholds = factor*thresholds
    thresholds = np.where(need_flip < 0, np.ceil(thresholds), np.floor(thresholds))
    # turn threshold into "number of 1s" (popcount) instead of signed sum
    if usePopCount and APrecision==1:
        thresholds = (fanin + thresholds)/2
    thresholds = thresholds.transpose(1,0).astype(np.int)

    # generating weights
    weights = weights * need_flip.reshape(-1,1,1,1)
    weights = quantize(weights, WPrecisions_int, WPrecisions_fract)
    if interleaveChannels:
        weights = np.moveaxis(weights, 1, -1)
    weights = weights.reshape((numOut, fanin))
    return (weights, thresholds)

# pull out data from a numpy archive containing layer parameters
# this should ideally be done using Lasagne, but this is simpler and works
class BNNWeightReader:
    def __init__(self, paramFile, interleaveChannels):
        self.paramDict = np.load(paramFile)
        self.currentParamInd = 0
        self.interleaveChannels = interleaveChannels
        self.numInterleaveChannels = 0

    def __getCurrent(self):
        ret =  self.paramDict["arr_" + str(self.currentParamInd)]
        self.currentParamInd += 1
        return ret.astype(np.float64)

    def readWeightsRaw(self):
        w = self.__getCurrent()
        return w

    def readBatchNormLayerRaw(self, read_bias=True):
        bias = self.__getCurrent() if read_bias else None
        beta = self.__getCurrent()
        gamma = self.__getCurrent()
        mean = self.__getCurrent()
        invstd = self.__getCurrent()
        return (bias, beta, gamma, mean, invstd)

    # read a fully connected layer plus batchnorm, binarize and convert to
    # positive threshold form, returning (bin weight matrix, thresholds)
    # the returned bin weight matrix has neurons along rows and is suitable
    # to be packed into BNN mems using BNNProcElemMem
    def readFCBNComplex(self, WPrecisions_fract, APrecisions_fract, IPrecisions_fract, \
        WPrecisions_int, APrecisions_int, IPrecisions_int, numThresBits=16, numThresIntBits=None):

        WPrecision = WPrecisions_fract + WPrecisions_int
        APrecision = APrecisions_fract + APrecisions_int
        IPrecision = IPrecisions_fract + IPrecisions_int
        weights = self.readWeightsRaw()
        (bias, beta, gamma, mean, invstd) = self.readBatchNormLayerRaw()

        if (WPrecision == 1) and (APrecision == 1) and (IPrecision == 1):
            (Wb, T) = makeFCBNComplex(weights, bias, beta, gamma, mean, invstd, WPrecisions_fract, APrecisions_fract, \
                WPrecisions_int, APrecisions_int, usePopCount=True)
        else:
            (Wb, T) = makeFCBNComplex(weights, bias, beta, gamma, mean, invstd, WPrecisions_fract, APrecisions_fract, \
                WPrecisions_int, APrecisions_int, numThresBits=numThresBits, numThresIntBits=numThresIntBits)

        # if the interleave flag is set, permute elements in each row
        if self.interleaveChannels and self.numInterleaveChannels != 0:
            print("Interleaving %d channels in fully connected layer..." % self.numInterleaveChannels)
            Wb = Wb.reshape(Wb.shape[0], self.numInterleaveChannels, -1)
            Wb = Wb.swapaxes(1,-1).reshape(Wb.shape[0], -1)
            # set interleave to zero once we go past this fc layer
            self.numInterleaveChannels = 0
        return (Wb, T)

    # read a fully connected layer without batchnorm and without using thresholds, 
    # returning bin weight matrix
    # the returned bin weight matrix has neurons along rows and is suitable
    # to be packed into BNN mems using BNNProcElemMem    
    def readFCBNComplex_no_thresholds(self, WPrecisions_fract, APrecisions_fract, IPrecisions_fract, \
        WPrecisions_int, APrecisions_int, IPrecisions_int, numThresBits=16, numThresIntBits=None):

        WPrecision = WPrecisions_fract + WPrecisions_int
        APrecision = APrecisions_fract + APrecisions_int
        IPrecision = IPrecisions_fract + IPrecisions_int

        weights = self.readWeightsRaw()
        (_, _, gamma, _, _) = self.readBatchNormLayerRaw(read_bias=False)

        #fake the batchnorm params to use same make functions below
        bias   = np.zeros(weights.shape[1])    
        beta   = np.zeros(weights.shape[1])
        #read gamma in case if it has a negative sign, we have to invert the weights
        gamma  = gamma*np.ones(weights.shape[1])
        mean   = np.ones(weights.shape[1])
        invstd = np.ones(weights.shape[1])

        if (WPrecision == 1) and (APrecision == 1) and (IPrecision == 1):
            (Wb, T) = makeFCBNComplex(weights, bias, beta, gamma, mean, invstd, WPrecisions_fract, APrecisions_fract, \
                WPrecisions_int, APrecisions_int, usePopCount=True)
        else:
            (Wb, T) = makeFCBNComplex(weights, bias, beta, gamma, mean, invstd, WPrecisions_fract, APrecisions_fract, \
                WPrecisions_int, APrecisions_int, numThresBits=numThresBits, numThresIntBits=numThresIntBits)

        # if the interleave flag is set, permute elements in each row
        if self.interleaveChannels and self.numInterleaveChannels != 0:
            print("Interleaving %d channels in fully connected layer..." % self.numInterleaveChannels)
            Wb = Wb.reshape(Wb.shape[0], self.numInterleaveChannels, -1)
            Wb = Wb.swapaxes(1,-1).reshape(Wb.shape[0], -1)
            # set interleave to zero once we go past this fc layer
            self.numInterleaveChannels = 0
        return (Wb, T)

    # read a convolutional layer plus batchnorm, binarize and convert to
    # positive threshold form, returning (bin weight matrix, thresholds)
    # the returned bin weight matrix  is suitable to be packed into BNN mems 
    def readConvBNComplex(self, WPrecisions_fract, APrecisions_fract, IPrecisions_fract, \
        WPrecisions_int, APrecisions_int, IPrecisions_int, usePopCount=True,numThresBits=16, numThresIntBits=None):
        
        weights = self.readWeightsRaw()
        (bias, beta, gamma, mean, invstd) = self.readBatchNormLayerRaw()
        # keep track of output channels for use in FC layer interleave
        self.numInterleaveChannels = weights.shape[0]
        (Wb, T) = makeConvBNComplex(weights, bias, beta, gamma, mean, invstd, self.interleaveChannels, \
            WPrecisions_fract, APrecisions_fract, IPrecisions_fract, WPrecisions_int, APrecisions_int, IPrecisions_int, \
            usePopCount=usePopCount, numThresBits=numThresBits, numThresIntBits=numThresIntBits)
        return (Wb, T)

# create a 2D array of zeroes for the PE memories    
def makeEmptyPEMems(numPE, memDepth, initVal):
    ret = np.full((numPE, memDepth), initVal)
    return ret

# ensure no non-binary weight values while packing
def ensureBinary(x):
    temp = np.where(x != 0, 1, x)
    temp = np.where(x != 1, 0, temp)
    if not np.array_equal(x,temp):
        raise Exception("Non-binary values found in BNN weight data")

# Encode the array as a single integer number
# The array contains all the values that has to be encoded
# in a single ap_uint.
def ArrayToAp_uints(array, precision, precFract=0):
    if precision == 1:
        ensureBinary(array)
        datatype = np.int64
    else:
        array = array * (1 << precFract)
        array = np.where(array < 0, array+(1 << precision), array).astype(np.uint64)
        datatype = np.uint64
    factor = 1 << precision*np.arange(array.shape[-1], dtype=datatype)
    val = array.dot(factor)
    return val


# pack one or several BNN layers into PE on-chip memories, and create
# initialization files (e.g. C++ initializer lists for HLS) from that
# note that no binarization or quantization is performed
# If numThresIntBits is not none, weights produced will be fixed point numbers.
class BNNProcElemMem:
    def __init__(self, numPE, numSIMD, weightMemDepth, thresMemDepth, \
        WPrecision_integer, APrecision_integer, IPrecision_integer,\
        WPrecision_fractional, APrecision_fractional, IPrecision_fractional, numThresBits=16, numThresIntBits=None):
        
        APrecision = APrecision_integer+APrecision_fractional
        WPrecision = WPrecision_integer+WPrecision_fractional
        IPrecision = IPrecision_integer+IPrecision_fractional
        self.numPE = numPE
        self.numSIMD = numSIMD
        if self.numSIMD*WPrecision > 64:
            raise Exception("SIMD*(WPrecision_integer+WPrecision_fractional) = {} which needs to be <= 64 for now. ".format(\
                self.numSIMD*WPrecision) +
                "If you wish extend the finnthesizer to support this, start with the old one " + \
                "(from previous commits).")
        
        self.numThresBits = numThresBits
        self.numThresIntBits = numThresIntBits
        self.APrecisionInt = APrecision_integer
        self.APrecisionFract = APrecision_fractional
        self.APrecision = APrecision
        self.numThresholds = 2**(APrecision - 1)
        self.IPrecision = IPrecision
        self.IPrecisionInt = IPrecision_integer
        self.IPrecisionFract = IPrecision_fractional
        self.WPrecision = WPrecision
        self.WPrecisionInt = WPrecision_integer
        self.WPrecisionFract = WPrecision_fractional
        self.weightMemDepth = weightMemDepth
        self.thresMemDepth = thresMemDepth
        # note that these memories are 2D: [PE index, mem index]
        if self.WPrecision==1:
            self.weightMem = makeEmptyPEMems(self.numPE, self.weightMemDepth, "1" * numSIMD)
        else:
            pad_word = ArrayToAp_uints(np.full((self.numSIMD),0), self.WPrecision)
            self.weightMem = makeEmptyPEMems(self.numPE, self.weightMemDepth, pad_word)
        self.thresMem = makeEmptyPEMems(self.numPE, self.thresMemDepth, 0)
        self.AccuOffset = 0
        self.neuronPad = []
        self.synapsePad = []
        self.layerSizes = []

    def __padMatrix(self, A, T, padW=0, padH=0):
        n = A.shape[0]
        s = A.shape[1]
        # ensure number of rows (neurons) is divisable by PE count
        padN = padH - n 
        # ensure number of cols (synapses per neuron) is divisable by SIMD width
        padS = padW - s
        # create padded version of matrix
        # use 1 bits to pad matrix, 0 bits to pad input
        const = 1 if self.WPrecision==1 else 0
        Ap = np.pad(A, ((0, padN), (0, padS)), 'constant', constant_values=const)
        # pad thresholds
        max_thres = pow(2, self.numThresBits) - 1
        Tp = np.pad(T, ((0, padN), (0, 0)), 'constant', constant_values=max_thres)
        if self.APrecision==1:
            Tp = Tp.reshape(-1,)
        # keep track of how much padding we added
        self.neuronPad += [padN]
        self.synapsePad += [padS]
        if (self.WPrecision==1 and self.APrecision==1 and self.IPrecision==1) or (self.WPrecision>=2):
            self.AccuOffset = 0
        else:       
            self.AccuOffset = padS
        return (Ap, Tp)

    def __updatePEMapping(self, A, T):
        # TODO also update threshold memories
        # should only be called internally, and on a matrix that is already padded
        n = A.shape[0]
        s = A.shape[1]
        if n % self.numPE != 0:
            raise Exception("Matrix height must be multiple of PE count")
        if s % self.numSIMD != 0:
            raise Exception("Matrix width must be multiple of SIMD width")
        if n != T.shape[0]:
            raise Exception("Number of neurons and thresholds do not match")
        # reshape and copy into PE memories
        neuronsPerPE = n // self.numPE
        synGroupsPerNeuron = s // self.numSIMD

        M = A.reshape((n, synGroupsPerNeuron, self.numSIMD))
        self.layerSizes += [(n,s)]

        M = ArrayToAp_uints(M, self.WPrecision, self.WPrecisionFract)
        
        tempw = np.split(M, neuronsPerPE, axis=0)
        tempw = np.asarray(tempw)
        tempw = np.split(tempw, synGroupsPerNeuron, axis=-1)
        tempw = np.asarray(tempw).swapaxes(0,2)
        tempw = tempw.reshape(tempw.shape[0], -1)        
        
        T = T - self.AccuOffset # We have to add the AccuOffset if padding is not transparent        
        tempt = np.split(T, neuronsPerPE, axis=0)
        tempt = np.array(tempt)
        tempt = tempt.swapaxes(0,1)        
        
        self.weightMem = tempw
        self.thresMem = tempt

        if self.numThresIntBits is None:
            # do saturation
            saturate_max = (2**(self.numThresBits-1))-1
            saturate_min = -(2**(self.numThresBits-1))
            self.thresMem = np.clip(self.thresMem, saturate_min, saturate_max)

    def addMatrix(self, W, T, padW=0, padH=0):
        # add padding
        if self.numThresIntBits is not None:
            T = T.astype(np.int)/(2.**(self.numThresBits - self.numThresIntBits))
        (Wp, Tp) = self.__padMatrix(W, T, padW, padH)
        # map to PE memories
        self.__updatePEMapping(Wp, Tp)

    def __makeHLSInit(self, x):
        if x == 0:
            return "0x0"
        else:
            return hex(np.uint64(x)) 

    # pack every word of the internal memory into a 64-bit integer and write
    # into a binary file
    def __wmem2bin(self, mem, fileName):  
        mem.astype(np.uint64).tofile(fileName)

    def __tmem2bin(self, mem, fileName):
        if self.numThresIntBits is None:
            if not np.array_equal(mem.astype(np.int), mem):
                print("WARNING: Cannot pack non-int values into binary threshold file.")
                print("The thresholds might be processed with wrong datatype. Check BNNProcElemMem \
                    arguments numThresBits and numThresIntBits to ensure correct fractional shift.")
        else:
            mem = mem * (1 << (self.numThresBits - self.numThresIntBits))
        mem.astype(np.int64).tofile(fileName)

    
    def createBinFiles(self, targetDir, prefix="", useThresholds=True):
        for pe in range(self.numPE):
            self.__wmem2bin(self.weightMem[pe], targetDir+"/"+prefix+"-"+str(pe)+"-weights.bin")
            if useThresholds:
                self.__tmem2bin(self.thresMem[pe], targetDir+"/"+prefix+"-"+str(pe)+"-thres.bin")

    # Finnthesizer HLS init files generation. Use these outputed header files for including params during bitstream generation
    def createHLSInitFiles(self, targetFile, varSuffix="", useThresholds=True):
        outFile = open(targetFile , "wt")
        if self.WPrecision==1:
            wMemType = "ap_uint<1>"
        elif self.WPrecisionFract==0:
            wMemType = "ap_int<"+str(self.WPrecisionInt)+">"
        else:
            wMemType = "ap_fixed<"+str(self.WPrecision)+", "+str(self.WPrecisionFract)+", AP_RND_ZERO, AP_WRAP>"
        if self.numThresIntBits is None:
            tMemType = "ap_int<"+str(self.numThresBits)+">"
        else:
            tMemType = "ap_fixed<"+str(self.numThresBits)+", "+str(self.numThresIntBits)+">"
        if self.APrecision==1:
            ActType = "ap_uint<1>"
        elif self.WPrecisionFract==0:
            ActType = "ap_int<"+str(self.APrecisionInt)+">"
        else:
            ActType = "ap_fixed<"+str(self.APrecision)+", "+str(self.APrecisionFract)+", AP_RND_ZERO, AP_WRAP>" 
        MinActVal = -1 # Minimum value of the output activations -> -1 if hardtanh, 0 with ReLu
        outFile.write("/*\nWeight and threshold memory initialization for Vivado HLS\n")
        outFile.write("PEs = %d, SIMD width = %d, threshold bits = %d\n" % (self.numPE, self.numSIMD, self.numThresBits))
        outFile.write("weight mem depth = %d, thres mem depth = %d\n" % (self.weightMemDepth, self.thresMemDepth))
        outFile.write("layer sizes (neurons, synapses per neuron): \n")
        outFile.writelines(["%s " % str(x) for x in self.layerSizes])
        outFile.write("\npadded neurons for each layer: \n")
        outFile.writelines(["%d " % x for x in self.neuronPad])
        outFile.write("\npadded synapses for each layer: \n")
        outFile.writelines(["%d " % x for x in self.synapsePad])
        outFile.write("\n*/\n\n")

        # write the weight memory init data
        # np.save("weightMem"+str(varSuffix)+".npy",self.weightMem)
        if self.WPrecision==1:
            outFile.write("static BinaryWeights<%d,%d,%d> weights%s= {\n{\n" % (self.numSIMD, self.numPE, self.weightMemDepth, varSuffix))
        else:
            outFile.write("static FixedPointWeights<%d,%s,%d,%d> weights%s= {\n{\n" % (self.numSIMD, wMemType, self.numPE, self.weightMemDepth, varSuffix))
        outFile.write(",".join(["{\n"+(",\n".join(map(self.__makeHLSInit, pe)))+"\n}" for pe in self.weightMem]))
        outFile.write("\n}\n};\n")

        # write the threshold memory init data
        if useThresholds:
            # np.save("threshMem"+str(varSuffix)+".npy",self.thresMem)
            if (self.numThresholds==1):
                outFile.write("static ThresholdsActivation<%d,%d,%d,%s,%s> threshs%s = {\n{\n" % (self.thresMemDepth, self.numPE, self.numThresholds, tMemType, ActType, varSuffix))
                outFile.write(",".join(["{\n"+(",\n".join(map(str, pe) ))+"\n}" for pe in self.thresMem]))
            else:
                outFile.write("static ThresholdsActivation<%d,%d,%d,%s,%s,%d> threshs%s = {\n{\n" % (self.thresMemDepth, self.numPE, self.numThresholds, tMemType, ActType, MinActVal, varSuffix))
                outFile.write(",".join(["{\n"+(",\n".join(["{\n"+",\n".join(map(str,nthresh))+"\n}" for nthresh in pe] ))+"\n}" for pe in self.thresMem]))
            outFile.write("\n}\n};\n")
        outFile.close()
