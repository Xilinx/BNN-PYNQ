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

import os
import sys
import finnthesizer as fth

if __name__ == "__main__":
    bnnRoot = "."
    npzFile = bnnRoot + "/mnist-1w-1a.npz"
    targetDirBin = bnnRoot + "/binparam-lfcW1A1-pynq"
    targetDirHLS = bnnRoot + "/binparam-lfcW1A1-pynq/hw"

    simdCounts = [64, 32, 64,  8]
    peCounts   = [32, 64, 32, 16]

    WeightsPrecisions_fractional    = [0, 0, 0, 0]
    ActivationPrecisions_fractional = [0, 0, 0, 0]
    InputPrecisions_fractional      = [0, 0, 0, 0]
    WeightsPrecisions_integer       = [1, 1, 1, 1]
    ActivationPrecisions_integer    = [1, 1, 1, 1]
    InputPrecisions_integer         = [1, 1, 1, 1]

    classes = map(lambda x: str(x), range(10))

    fth.convertFCNetwork(npzFile, targetDirBin, targetDirHLS, simdCounts, peCounts, WeightsPrecisions_fractional, ActivationPrecisions_fractional, InputPrecisions_fractional, WeightsPrecisions_integer, ActivationPrecisions_integer, InputPrecisions_integer)

    with open(targetDirBin + "/classes.txt", "w") as f:
        f.write("\n".join(classes))

