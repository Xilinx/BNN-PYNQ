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
 * @file slidingwindow.h
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file describe the input generator used in the convolutional layer to 
 * output the inputfeature map to perform the matrix-vector reduction of a 
 * convolution 
 * 
 *
 *****************************************************************************/

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))
// II=1 sliding window generator, with common iteration space
template<unsigned int ConvKernelDim, unsigned int IFMChannels,
		unsigned int IFMDim, unsigned int OFMDim, unsigned int InpWidth=1, unsigned int PadDim=0>
void StreamingConvolutionInputGenerator_Batch(
		stream<ap_uint<IFMChannels*InpWidth> > & in,
		stream<ap_uint<IFMChannels*InpWidth> > & out,
		const unsigned int numReps,
		ap_uint<IFMChannels*InpWidth> padValue = 0) {

	constexpr unsigned int IFMChanChunk = IFMChannels*InpWidth;
  constexpr unsigned int IFMPadDim = IFMDim + 2*PadDim;
  constexpr int IFMLoopBound = IFMDim + PadDim;
  ap_uint<IFMChanChunk> inputBuf[IFMPadDim * IFMPadDim];
#pragma HLS_RESOURCE variable inputBuf core=RAM_S2P_BRAM

    const unsigned int additional_lines = (IFMPadDim*IFMPadDim)/(OFMDim * ConvKernelDim * ConvKernelDim);
    const unsigned int Initial_lines =  ((IFMPadDim) < ((OFMDim * ConvKernelDim * ConvKernelDim)) ? (ConvKernelDim+1) : (ConvKernelDim + additional_lines - IFMDim));
    const unsigned int Initial_buffer = MIN(Initial_lines * (IFMPadDim),IFMPadDim * IFMPadDim-1);
    const unsigned int baseIter = Initial_buffer
			+ (OFMDim * OFMDim * ConvKernelDim * ConvKernelDim);


	unsigned int inp = 0, oy = 0, ox = 0, ky = 0, kx = 0;
	int inp_i = -PadDim, inp_j = -PadDim;
	for (unsigned int i = 0; i < baseIter * numReps; i++) {
#pragma HLS PIPELINE II=1
	if (inp < IFMPadDim * IFMPadDim) {
      		ap_uint<IFMChanChunk> inElem;
     		if ((inp_i < 0) || (inp_j < 0) || (inp_i >= IFMDim) || (inp_j >= IFMDim)) {
          		inElem = padValue;
     		}
     		else {
        		inElem = in.read();
      		}
		inputBuf[inp] = inElem;
		inp++;
		inp_j++;
		if(inp_j == IFMLoopBound) {
		  inp_j = -PadDim;
		  inp_i++;
		  if(inp_i == IFMLoopBound) {
		    inp_i = -PadDim;
		  }
		}
		} 
		if (inp > Initial_buffer)
		{
			unsigned int input_base = oy * IFMPadDim + ox;
			unsigned int input_ind = input_base + ky * IFMPadDim + kx;
			ap_uint<IFMChanChunk> inElem = inputBuf[input_ind];
			out.write(inElem);
			kx++;
			if (kx == ConvKernelDim) {
				kx = 0;
				ky++;
				if (ky == ConvKernelDim) {
					ky = 0;
					ox++;
					if (ox == OFMDim) {
						ox = 0;
						oy++;
						if (oy == OFMDim) {
							oy = 0;
							inp = 0;
						}
					}
				}
			}
		}
	}
}
