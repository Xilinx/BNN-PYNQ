#BSD 3-Clause License
#=======
#
#Copyright (c) 2018, Xilinx
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

import os
import sys
from finnthesizer import *

if __name__ == "__main__":
    bnnRoot = "."
    npzFile = bnnRoot + "/gtsrb_parameters.npz"
    targetDirBin = bnnRoot + "/binparam-cnv-pynq"
    
    peCounts = [16, 32, 16, 16, 4, 1, 1, 1, 4]
    simdCounts = [3, 32, 32, 32, 32, 32, 4, 8, 1]
    
    classes =   [
                '20 Km/h',
                '30 Km/h',
                '50 Km/h',
                '60 Km/h',
                '70 Km/h',
                '80 Km/h',
                'End 80 Km/h',
                '100 Km/h',
                '120 Km/h',
                'No overtaking',
                'No overtaking for large trucks',
                'Priority crossroad',
                'Priority road',
                'Give way',
                'Stop',
                'No vehicles',
                'Prohibited for vehicles with a permitted gross weight over 3.5t including their trailers, and for tractors except passenger cars and buses',
                'No entry for vehicular traffic',
                'Danger Ahead',
                'Bend to left',
                'Bend to right',
                'Double bend (first to left)',
                'Uneven road',
                'Road slippery when wet or dirty',
                'Road narrows (right)',
                'Road works',
                'Traffic signals',
                'Pedestrians in road ahead',
                'Children crossing ahead',
                'Bicycles prohibited',
                'Risk of snow or ice',
                'Wild animals',
                'End of all speed and overtaking restrictions',
                'Turn right ahead',
                'Turn left ahead',
                'Ahead only',
                'Ahead or right only',
                'Ahead or left only',
                'Pass by on right',
                'Pass by on left',
                'Roundabout',
                'End of no-overtaking zone',
                'End of no-overtaking zone for vehicles with a permitted gross weight over 3.5t including their trailers, and for tractors except passenger cars and buses',
                'Not a roadsign'
                ]
    
    if not os.path.exists(targetDirBin):
      os.mkdir(targetDirBin)
      
    rHW = BNNWeightReader(npzFile, True)
    
    # TODO:
    # - generalize and move into library
    # - spit out config header
    # - add param generation for SVHN
    
    # process convolutional layers
    for convl in range(0, 6):
      peCount = peCounts[convl]
      simdCount = simdCounts[convl]
      print "Using peCount = %d simdCount = %d for engine %d" % (peCount, simdCount, convl)
      if convl == 0:
        # use fixed point weights for the first layer
        (w,t) = rHW.readConvBNComplex(usePopCount=False)
        # compute the padded width and height
        paddedH = padTo(w.shape[0], peCount)
        paddedW = padTo(w.shape[1], simdCount)
        # compute memory needed for weights and thresholds
        neededWMem = (paddedW * paddedH) / (simdCount * peCount)
        neededTMem = paddedH / peCount
        print "Layer %d: %d x %d" % (convl, paddedH, paddedW)
        print "WMem = %d TMem = %d" % (neededWMem, neededTMem)
        m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem, numThresBits=24, numThresIntBits=16)
        m.addMatrix(w,t)
        m.createBinFiles(targetDirBin, str(convl))
      else:
        # regular binarized layer
        (w,t) = rHW.readConvBNComplex()
        # compute the padded width and height
        paddedH = padTo(w.shape[0], peCount)
        paddedW = padTo(w.shape[1], simdCount)
        # compute memory needed for weights and thresholds
        neededWMem = (paddedW * paddedH) / (simdCount * peCount)
        neededTMem = paddedH / peCount
        print "Layer %d: %d x %d" % (convl, paddedH, paddedW)
        print "WMem = %d TMem = %d" % (neededWMem, neededTMem)
        m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem)
        m.addMatrix(w,t)
        m.createBinFiles(targetDirBin, str(convl))
    
    # process fully-connected layers
    for fcl in range(6,9):
      peCount = peCounts[fcl]
      simdCount = simdCounts[fcl]
      print "Using peCount = %d simdCount = %d for engine %d" % (peCount, simdCount, fcl)
      (w,t) =  rHW.readFCBNComplex()
      # compute the padded width and height
      paddedH = padTo(w.shape[0], peCount)
      paddedW = padTo(w.shape[1], simdCount)
      # compute memory needed for weights and thresholds
      neededWMem = (paddedW * paddedH) / (simdCount * peCount)
      neededTMem = paddedH / peCount
      print "Layer %d: %d x %d" % (fcl, paddedH, paddedW)
      print "WMem = %d TMem = %d" % (neededWMem, neededTMem)
      m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem)
      m.addMatrix(w,t)
      m.createBinFiles(targetDirBin, str(fcl))
    
    with open(targetDirBin + "/classes.txt", "w") as f:
        f.write("\n".join(classes))

