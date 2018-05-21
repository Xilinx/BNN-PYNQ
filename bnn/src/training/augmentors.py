#BSD 3-Clause License
#=======
#
#Copyright (c) 2018, Xilinx Inc.
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

import numpy as np
import scipy.ndimage as ndimg

def linear_rotations(X, Y, angles, original=True):
    Xo = np.copy(X)
    Yo = np.copy(Y)
    i = 0
    for angle in angles:
        Xtmp = ndimg.rotate(Xo, angle, prefilter=False, mode='nearest', axes=(2,3), reshape=False)
        Ytmp = np.copy(Yo)
        if not original and i == 0:
            X = Xtmp
            Y = Ytmp
        else:
            X = np.append(X, Xtmp, axis=0)
            Y = np.append(Y, Ytmp, axis=0)
        i += 1
    return (X, Y)

def random_rotations(X, Y, rnd_range, factor, extend=True):
    if extend:
        angles = np.random.uniform(rnd_range[0], rnd_range[1], size=factor)
        return linear_rotations(X, Y, angles)
    else:
        X = np.copy(X)
        Y = np.copy(Y)
        N = len(Y)
        angles = np.random.uniform(rnd_range[0], rnd_range[1], size=(N))
        X = np.array(map(lambda v: ndimg.rotate(v[0], v[1], prefilter=False, mode='nearest', axes=(1,2), reshape=False), zip(X,angles)))
        return (X, Y)

def adjusted_crop(X, Y, offsets, size):
    Xo = np.copy(X)
    Yo = np.copy(Y)
    w = size[0]
    h = size[1]
    i = 0
    for offset in offsets:
        wo=offset[0]
        ho=offset[1]
        Xtmp = np.copy(Xo[:,:,wo:wo+w,ho:ho+h])
        Ytmp = np.copy(Yo)
        if i == 0:
            X = Xtmp
            Y = Ytmp
        else:
            X = np.append(X, Xtmp, axis=0)
            Y = np.append(Y, Ytmp, axis=0)
        i += 1
    return (X, Y)

def random_crop(X, Y, rnd_range, factor, size, extend=True):
    if extend:
        offsets = np.random.randint(rnd_range[0], size=(factor,1))
        offsets = np.append(offsets, np.random.randint(rnd_range[1], size=(factor,1)), axis=1)
        return adjusted_crop(X, Y, offsets, size)
    else:
        N = len(Y)
        w = size[0]
        h = size[1]
        wos = np.random.randint(rnd_range[0], size=(N))
        hos = np.random.randint(rnd_range[1], size=(N))
        X = np.array(map(lambda v: v[0][:,v[1]:v[1]+w,v[2]:v[2]+h], zip(X,wos,hos)))
        return (X, Y)

