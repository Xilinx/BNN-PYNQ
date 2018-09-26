/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
 
/******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file maxpool.h
 *
 *  Library of templated HLS functions for BNN deployment. 
 *  This file implement the BNN maxpool layer.
 *
 ******************************************************************************/

#ifndef MAXPOOL_H
#define MAXPOOL_H
 
#include <limits>
 
template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels>
void StreamingMaxPool(stream<ap_uint<NumChannels> > & in,
		stream<ap_uint<NumChannels> > & out) {
  CASSERT_DATAFLOW(ImgDim % PoolDim == 0);
  // need buffer space for a single maxpooled row of the image
  ap_uint<NumChannels> buf[ImgDim / PoolDim];
  for(unsigned int i = 0; i < ImgDim / PoolDim; i++) {
#pragma HLS UNROLL
    buf[i] = 0;
  }

  for (unsigned int yp = 0; yp < ImgDim / PoolDim; yp++) {
    for (unsigned int ky = 0; ky < PoolDim; ky++) {
      for (unsigned int xp = 0; xp < ImgDim / PoolDim; xp++) {
#pragma HLS PIPELINE II=1
        ap_uint<NumChannels> acc = 0;
        for (unsigned int kx = 0; kx < PoolDim; kx++) {
          acc = acc | in.read();
        }
        // pool with old value in row buffer
        buf[xp] |= acc;
      }
    }
	for (unsigned int outpix = 0; outpix < ImgDim / PoolDim; outpix++) {
#pragma HLS PIPELINE II=1
      out.write(buf[outpix]);
      // get buffer ready for next use
      buf[outpix] = 0;
    }
  }
}

// calling 1-image maxpool in a loop works well enough for now
template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels>
void StreamingMaxPool_Batch(stream<ap_uint<NumChannels> > & in,
		stream<ap_uint<NumChannels> > & out, unsigned int numReps) {
  for (unsigned int rep = 0; rep < numReps; rep++) {
    StreamingMaxPool<ImgDim, PoolDim, NumChannels>(in, out);
  }
}

template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels, typename ActType, int min_value, 
		int StreamW // safely deducible (stream width must be int though!)
		>
void StreamingMaxPool_Precision(stream<ap_uint<StreamW> > & in,
		stream<ap_uint<StreamW> > & out) {
  CASSERT_DATAFLOW(ImgDim % PoolDim == 0);
  // need buffer space for a single maxpooled row of the image
  ActType buf[ImgDim / PoolDim][NumChannels];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=2
  for(unsigned int i = 0; i < ImgDim / PoolDim; i++) {
    for(unsigned int ch = 0; ch<NumChannels; ch++){
#pragma HLS UNROLL
      buf[i][ch] = min_value; //std::numeric_limits<ActType>::min();
    }
  }
  ap_uint<StreamW> inputData,outputData;
  for (unsigned int yp = 0; yp < ImgDim / PoolDim; yp++) {
    for (unsigned int ky = 0; ky < PoolDim; ky++) {
      for (unsigned int xp = 0; xp < ImgDim / PoolDim; xp++) {
        // Change to comparator	
        for (unsigned int kx = 0; kx < PoolDim; kx++) {
#pragma HLS PIPELINE II=1
          inputData = in.read();
          for(unsigned int ch = 0; ch<NumChannels; ch++){
#pragma HLS UNROLL						
            unsigned int lowBit = ch * ActType::width;
            unsigned int highBit = (ch+1) * ActType::width -1;
            ActType channeldata = inputData(highBit, lowBit);					
            ActType oldMax = buf[xp][ch];				
            if(channeldata > oldMax){
              buf[xp][ch] = channeldata;
            }
          }
        }
      }
    }
    for (unsigned int outpix = 0; outpix < ImgDim / PoolDim; outpix++) {
      for(unsigned int ch = 0; ch < NumChannels; ch++){
#pragma HLS UNROLL
        unsigned int lowBit = ch * ActType::width;
        unsigned int highBit = (ch+1) * ActType::width -1;	
        outputData(highBit, lowBit) = buf[outpix][ch];
        // get buffer ready for next use
        buf[outpix][ch] = min_value;
      }
      out.write(outputData);
    }
  }
}

// calling 1-image maxpool in a loop works well enough for now
template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels, typename ActType, int min_value, 
        int InStreamW, int OutStreamW  // safely deducible (stream width must be int though!)
		>
void StreamingMaxPool_Precision_Batch(stream<ap_uint<InStreamW> > & in,
		stream<ap_uint<OutStreamW> > & out, unsigned int numReps) {
#pragma HLS INLINE
  unsigned const  InpPerImage = ImgDim*ImgDim*NumChannels*ActType::width/InStreamW ;
  unsigned const  OutPerImage = ImgDim*ImgDim / (PoolDim*PoolDim);
  WidthAdjustedInputStream <InStreamW, NumChannels*ActType::width, InpPerImage>  wa_in (in,  numReps);
  WidthAdjustedOutputStream<NumChannels*ActType::width,  OutStreamW, OutPerImage>  wa_out(out, numReps);
  for (unsigned int rep = 0; rep < numReps; rep++) {
    StreamingMaxPool_Precision<ImgDim, PoolDim, NumChannels, ActType, min_value>
      (static_cast<hls::stream<ap_uint<NumChannels*ActType::width>>&>(wa_in), 
      static_cast<hls::stream<ap_uint<NumChannels*ActType::width>>&>(wa_out));
  }
}

#endif
