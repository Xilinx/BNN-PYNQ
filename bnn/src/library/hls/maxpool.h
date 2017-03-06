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
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file maxpool.h
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file implement the BNN maxpool layer 
 * 
 *
 *****************************************************************************/

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
