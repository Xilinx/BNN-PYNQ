#BSD 3-Clause License
#=======
#
#Copyright (c) 2017, Xilinx
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import sys

# convenience function to prepare a fully-connected BNN for FINN
# with given number of (SIMD, PE) per layer
def convertFCNetwork(npzFile, targetDirBin, simdCounts, peCounts):
  numLayers = len(simdCounts)
  if not os.path.exists(targetDirBin):
    os.mkdir(targetDirBin)

  # instantiate the weight reader, note how interleaveChannels=False for a 
  # fully connected network
  r = BNNWeightReader(npzFile, False)

  config = ""
  for l in range(numLayers):
    simdCount = simdCounts[l]
    peCount = peCounts[l]
    # read out weights and thresholds
    (w,t) = r.readFCBNComplex()
    # compute the padded width and height
    paddedH = padTo(w.shape[0], peCount)
    paddedW = padTo(w.shape[1], simdCount)
    # compute memory needed for weights and thresholds
    neededWMem = (paddedW * paddedH) / (simdCount * peCount)
    neededTMem = paddedH / peCount
    
    print "Layer %d: %d x %d, SIMD = %d, PE = %d" % (l, paddedH, paddedW, simdCount, peCount)
    print "WMem = %d TMem = %d" % (neededWMem, neededTMem)
    config += (printFCDefines("L%d" % l, simdCount, peCount, neededWMem, neededTMem, paddedW, paddedH)) + "\n"
    # instantiate PE memory generator
    m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem)
    # pack parameters into PE memory
    m.addMatrix(w,t)
    # create HLS weight init files for initializing memory contents directly
    # while generating the bitstream
    #m.createHLSInitFiles(targetDirHLS + "/memdata-" + str(l) + ".h", str(l))
    # create binary weight files -- useful for runtime initialization since
    # HLS might freeze / not work for very large header files
    # note that it will still be necessary to declare the PE memories in 
    m.createBinFiles(targetDirBin, str(l))
    # create parameter files for tiny-cnn
    #makeBinParamsForSoftware(targetDirSW, str(l), (w,t))

  print "Config header file:\n\n"
  print config  

# return HW config string as C #define's for a FC layer
def printFCDefines(prefix, simd, pe, wmem, tmem, mw, mh):
  config = ""
  config += "\n" + "#define %s_SIMD %d" % (prefix, simd)
  config += "\n" + "#define %s_PE %d" % (prefix, pe)
  config += "\n" + "#define %s_WMEM %d" % (prefix, wmem)
  config += "\n" + "#define %s_TMEM %d" % (prefix, tmem)
  config += "\n" + "#define %s_MW %d" % (prefix, mw)
  config += "\n" + "#define %s_MH %d" % (prefix, mh)
  return config
  
# return val to nearest multiple of pad
def padTo(val, pad):
  rem = val % pad
  return val if rem == 0 else (val + pad - rem)

# the binarization function, basically the sign encoded as 1 for positive and
# 0 for negative
def binarize(w):
    return 1 if w >=0 else 0

# convert a fully connected binarized layer plus batch normalization into 
# the simplified form (binary weight and positive threshold)
# note that the neurons are assumed to be in the columns of the weight
# matrix
def makeFCBNComplex(weights, beta, gamma, mean, invstd, use_rowmajor=False, usePopCount=True):
  ins = weights.shape[0]
  outs = weights.shape[1]
  print "Extracting FCBN complex, ins = %d outs = %d" % (ins, outs)
  # we'll fill in the binarized weights and thresholds iteratively
  w_bin = range(ins*outs)
  thresholds = range(outs)
  for neuron in range(outs):
    # compute a preliminary threshold from the batchnorm parameters
    thres = mean[neuron] - (beta[neuron] / (gamma[neuron]*invstd[neuron]))
    need_flip = 0
    # ensure all neurons activate on the "positive" side, so we can use
    # greater-than-threshold activation
    if gamma[neuron]*invstd[neuron] < 0:
        need_flip = 1
        thres = -thres
    # turn threshold into "number of 1s" (popcount) instead of signed sum
    if usePopCount:
        thresholds[neuron] = int((ins + thres) / 2)
    else:
        thresholds[neuron] = thres
    # binarize the synapses
    for synapse in range(ins):
      # note how we change from col major to row major if requested
      dest_ind = neuron*ins+synapse if use_rowmajor else synapse*outs+neuron
      if need_flip:
        w_bin[dest_ind] = binarize(-weights[synapse][neuron])
      else:
        w_bin[dest_ind] = binarize(weights[synapse][neuron])
  # reshape the output as desired
  if use_rowmajor:
    w_bin = np.asarray(w_bin).reshape((outs, ins))
  else:
    w_bin = np.asarray(w_bin).reshape((ins, outs))
    
  return (w_bin, thresholds)
  
# binarize and pack convolutional layer weights into a matrix and compute
# thresholds from the conv bias and batchnorm parameters
def makeConvBNComplex(weights, bias, beta, gamma, mean, invstd, interleaveChannels, usePopCount=True):
  numOut = weights.shape[0]
  numIn = weights.shape[1]
  k = weights.shape[2]
  if(k != weights.shape[3]):
    raise "Nonsymmetric conv kernels are not yet supported"
  print "Extracting conv-BN complex, OFM=%d IFM=%d k=%d" % (numOut, numIn, k)
  # the fanin is used to ensure positive-only threshold
  fanin = numIn * k * k
  w_bin = range(numOut * numIn * k * k)
  # one threshold per output channel
  thresholds = range(numOut)
  dest_ind = 0
  # we'll fill in the binarized weights and thresholds iteratively
  for neuron in range(numOut):
    # compute a preliminary threshold from the batchnorm parameters,
    # subtracting the conv bias from the batchnorm mean
    thres = (mean[neuron] - bias[neuron]) - (beta[neuron] / (gamma[neuron]*invstd[neuron]))
    need_flip = 0
    # ensure all neurons activate on the "positive" side, so we can use
    # greater-than-threshold activation
    if gamma[neuron]*invstd[neuron] < 0:
        need_flip = 1
        thres = -thres
    # turn threshold into "number of 1s" (popcount) instead of signed sum
    if usePopCount:
        thresholds[neuron] = int((fanin + thres) / 2)
    else:
        thresholds[neuron] = thres
    # go through each weight of each convolutional kernel
    if interleaveChannels:
      for ky in range(k):
        for kx in range(k):
          for ifm in range(numIn):
            f = -1 if need_flip else +1
            w_bin[dest_ind] = binarize(f*weights[neuron][ifm][ky][kx])
            dest_ind += 1
    else:
      for ifm in range(numIn):
        for ky in range(k):
          for kx in range(k):
            f = -1 if need_flip else +1
            w_bin[dest_ind] = binarize(f*weights[neuron][ifm][ky][kx])
            dest_ind += 1
          
  # reshape the output as desired
  w_bin = np.asarray(w_bin).reshape((numOut, fanin))
  return (w_bin, thresholds)
  

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
    return ret
    
  def readFCLayerRaw(self):
    w = self.__getCurrent()
    b = self.__getCurrent()
    return (w, b)
    
  def readConvLayerRaw(self):
    w = self.__getCurrent()
    b = self.__getCurrent()
    return (w, b)
    
  def readBatchNormLayerRaw(self):
    beta = self.__getCurrent()
    gamma = self.__getCurrent()
    mean = self.__getCurrent()
    invstd = self.__getCurrent()
    return (beta, gamma, mean, invstd)
    
  # read a fully connected layer plus batchnorm, binarize and convert to
  # positive threshold form, returning (bin weight matrix, thresholds)
  # the returned bin weight matrix has neurons along rows and is suitable
  # to be packed into BNN mems using BNNProcElemMem
  def readFCBNComplex(self):
    (w,b) = self.readFCLayerRaw()
    (beta, gamma, mean, invstd) = self.readBatchNormLayerRaw()
    (Wb, T) = makeFCBNComplex(w, beta, gamma, mean, invstd, use_rowmajor=True)
    # if the interleave flag is set, permute elements in each row
    if self.interleaveChannels and self.numInterleaveChannels != 0:
      print "Interleaving %d channels in fully connected layer..." % self.numInterleaveChannels
      pixPerChan = Wb.shape[1] / self.numInterleaveChannels
      Wb_perm = np.zeros(Wb.shape, dtype=np.int8)
      for r in range(Wb.shape[0]):
        for chan in range(self.numInterleaveChannels):
          for cpix in range(pixPerChan):
            Wb_perm[r][cpix*self.numInterleaveChannels + chan] = Wb[r][chan*pixPerChan + cpix]
      Wb = Wb_perm
      # set interleave to zero once we go past this fc layer
      self.numInterleaveChannels = 0
      
    return (Wb, T)
    
  # read a convolutional layer plus batchnorm, binarize and convert to
  # positive threshold form, returning (bin weight matrix, thresholds)
  # the returned bin weight matrix  is suitable to be packed into BNN mems 
  def readConvBNComplex(self, usePopCount=True):
    (w,b) = self.readConvLayerRaw()
    # keep track of output channels for use in FC layer interleave
    self.numInterleaveChannels = w.shape[0]
    (beta, gamma, mean, invstd) = self.readBatchNormLayerRaw()
    (Wb, T) = makeConvBNComplex(w, b, beta, gamma, mean, invstd, self.interleaveChannels, usePopCount=usePopCount)
    return (Wb, T)

# create a 2D array of zeroes for the PE memories    
def makeEmptyPEMems(numPE, memDepth, initVal):
  ret = []
  
  for i in range(numPE):
    ret += [ [initVal for i in range(memDepth)] ]
  return ret

# ensure no non-binary weight values while packing
def ensureBinary(x):
  for i in x:
    if i != 0 and i != 1:
      raise "Non-binary values found in BNN weight data"
  
# turn a binary array into a string representation with MSB on the left
# e.g. [A, B, C, D] becomes "DCBA"
def binArrayToString(x):
  ensureBinary(x)
  return reduce(lambda x,y: str(x)+str(y), np.flipud(x), "")

# pack one or several BNN layers into PE on-chip memories, and create
# initialization files (e.g. C++ initializer lists for HLS) from that
# note that no binarization or quantization is performed
# If numThresIntBits is not none, weights produced will be fixed point numbers.
class BNNProcElemMem:
  def __init__(self, numPE, numSIMD, weightMemDepth, thresMemDepth, numThresBits=16, numThresIntBits=None):
    self.numPE = numPE
    self.numSIMD = numSIMD
    self.numThresBits = numThresBits
    self.numThresIntBits = numThresIntBits
    self.weightMemDepth = weightMemDepth
    self.thresMemDepth = thresMemDepth
    self.weightMemHead = [0 for i in range(numPE)]
    self.thresMemHead = [0 for i in range(numPE)]
    # note that these memories are 2D: [PE index, mem index]
    self.weightMem = makeEmptyPEMems(self.numPE, self.weightMemDepth, "1" * numSIMD)
    self.thresMem = makeEmptyPEMems(self.numPE, self.thresMemDepth, 0)
    
    self.neuronPad = []
    self.synapsePad = []
    self.layerSizes = []
    self.layerHeadsW = []
    self.layerHeadsT = []
 
  def __padMatrix(self, A, T):
    n = A.shape[0]
    s = A.shape[1]
    # ensure number of rows (neurons) is divisable by PE count
    padN = self.numPE - (n % self.numPE) if n % self.numPE != 0 else 0
    # ensure number of cols (synapses per neuron) is divisable by SIMD width
    padS = self.numSIMD - (s % self.numSIMD) if s % self.numSIMD != 0 else 0
    # create padded version of matrix
    # use 1 bits to pad matrix, 0 bits to pad input
    Ap = np.pad(A, ((0, padN), (0, padS)), 'constant', constant_values=1 )
    # pad thresholds
    max_thres = pow(2, self.numThresBits) - 1
    Tp = np.pad(T, ((0, padN)), 'constant', constant_values=max_thres)
    # keep track of how much padding we added
    self.neuronPad += [padN]
    self.synapsePad += [padS]
    return (Ap, Tp)
    
  def __updatePEMapping(self, A, T):
    # TODO also update threshold memories
    # should only be called internally, and on a matrix that is already padded
    n = A.shape[0]
    s = A.shape[1]
    if n % self.numPE != 0:
      raise "Matrix height must be multiple of PE count"
    if s % self.numSIMD != 0:
      raise "Matrix width must be multiple of SIMD width"
    if n != T.shape[0]:
      raise "Number of neurons and thresholds do not match"
    # reshape and copy into PE memories
    neuronsPerPE = n / self.numPE
    synGroupsPerNeuron = s / self.numSIMD
    # TODO check that there is enough room in the PE memory
    self.layerHeadsW += [ self.weightMemHead[0] ]
    self.layerHeadsT += [ self.thresMemHead[0] ]
    
    M=A.reshape((n, synGroupsPerNeuron, self.numSIMD))
    self.layerSizes += [(n,s)]
    
    for i in range(n):
      # interleave matrix rows between different PEs
      targetPE = i % self.numPE
      targetBase = self.weightMemHead[targetPE]
      self.thresMem[targetPE][self.thresMemHead[targetPE]] = T[i]
      
      for j in range(synGroupsPerNeuron):
        self.weightMem[targetPE][targetBase+j] = binArrayToString(M[i][j])
      # update the memory head pointers for the target PE
      self.weightMemHead[targetPE] += synGroupsPerNeuron
      self.thresMemHead[targetPE] += 1
           
  def addMatrix(self, W, T):
    # add padding
    if self.numThresIntBits is None:
        (Wp, Tp) = self.__padMatrix(W, T)
    else: # Convert thresholds to ints before updating the PE mapping.
        Ti = map(lambda x: int(x*2**(self.numThresBits - self.numThresIntBits)), T)
        (Wp, Tp) = self.__padMatrix(W, Ti)
    # map to PE memories
    self.__updatePEMapping(Wp, Tp)
    
  def __makeHLSInit(self, x):
    if x == 0:
      return "0"
    else:
      return "ap_uint<"+str(self.numSIMD)+">(\""+ x +"\", 2)"
      
  # pack every word of the internal memory into a 64-bit integer and write
  # into a binary file
  # TODo support sizes > 64 bits. use bitstring module? 
  def __mem2bin(self, mem, fileName, isBinaryString, fmt="Q"):
    import struct
    outFile = open(fileName, "wb")
    for memInd in range(len(mem)):
      if isBinaryString:
        if len(mem[memInd]) > 64:
          raise "SIMD width needs to be max 64 bits for binary packing for now"
        outFile.write(struct.pack(fmt, int(mem[memInd], 2)))
      else:
        if fmt == "Q":
          if mem[memInd] <= 0:
            print "Warning: Zero or negative (val=%d) threshold detected." % mem[memInd]
          mem[memInd] = max(0, mem[memInd])
        outFile.write(struct.pack(fmt, mem[memInd]))
    outFile.close()
    
  def createBinFiles(self, targetDir, prefix=""):
    if self.numThresIntBits is None:
      fmt = 'Q'
    else:
      fmt = 'q'
    for pe in range(self.numPE):
      # always use unsigned long long ('Q') for packing the weights, this is raw 64-bit data
      # that does not really need a sign bit. otherwise, if the most sig, bit is 1, Python
      # will complain about "integer out of range".
      self.__mem2bin(self.weightMem[pe], targetDir+"/"+prefix+"-"+str(pe)+"-weights.bin", True, 'Q')
      self.__mem2bin(self.thresMem[pe], targetDir+"/"+prefix+"-"+str(pe)+"-thres.bin", False, fmt)
    
  def createHLSInitFiles(self, targetFile, varSuffix=""):
    outFile = open(targetFile , "at")
    wMemType = "ap_uint<"+str(self.numSIMD)+">"
    if self.numThresIntBits is None:
        tMemType = "ap_uint<"+str(self.numThresBits)+">"
    else:
        tMemType = "ap_fixed<"+str(self.numThresBits)+", "+str(self.numThresIntBits)+">"
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
    outFile.write("const unsigned int matrixH"+varSuffix+"[] = {%s};\n" % ", ".join(map(lambda x: str(x[0]), self.layerSizes)))
    outFile.write("const unsigned int matrixW"+varSuffix+"[] = {%s};\n" % ", ".join(map(lambda x: str(x[1]), self.layerSizes)))
    outFile.write("const unsigned int layerStartW"+varSuffix+"[] = {%s};\n" % ", ".join(map(str, self.layerHeadsW)))
    outFile.write("const unsigned int layerStartT"+varSuffix+"[] = {%s};\n\n" % ", ".join(map(str, self.layerHeadsT)))
    
    # write the weight memory init data
    outFile.write("const %s weightMem%s[%d][%d] = {\n" % (wMemType, varSuffix, self.numPE, self.weightMemDepth))
    outFile.write(",".join(map(lambda pe:"{\n"+(",\n".join(map(self.__makeHLSInit, pe)))+"\n}", self.weightMem)))
    outFile.write("\n};\n")
    # write the threshold memory init data
    outFile.write("const %s thresMem%s[%d][%d] = {\n" % (tMemType, varSuffix, self.numPE, self.thresMemDepth))
      
    outFile.write(",".join(map(lambda pe:"{\n"+(",\n".join(map(str, pe) ))+"\n}", self.thresMem)))
    outFile.write("\n};\n")
    outFile.close()
    
# convenience function for turning weight-threshold pairs into binary parameter
# files that are suitable for loading into tiny-cnn software layers.
# this is achieved by packing the parameters into an MV engine with PE = 1 and
# SIMD = 1 so no padding and no packing occurs.
def makeBinParamsForSoftware(targetDir, prefix, (weights, thresholds)):
    m = BNNProcElemMem(1, 1, weights.shape[0]*weights.shape[1], weights.shape[0])
    m.addMatrix(weights, thresholds)
    m.createBinFiles(targetDir, prefix)

