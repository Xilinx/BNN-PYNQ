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
 * @file fclayer.h
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file lists a set of convenience funtions used to implement fully 
 * connected layers
 * 
 *
 *****************************************************************************/

// helper function for fully connected layers
// instantiates matrix vector unit plus data width converters
template<unsigned int InStreamW, unsigned int OutStreamW,
		unsigned int SIMDWidth, unsigned int PECount,
		unsigned int PopCountWidth, unsigned int MatrixW, unsigned int MatrixH,
		unsigned int WMemCount, unsigned int TMemCount>
void StreamingFCLayer_Batch(stream<ap_uint<InStreamW> > & in,
		stream<ap_uint<OutStreamW> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_uint<PopCountWidth> thresMem[PECount][TMemCount],
		const unsigned int numReps) {
#pragma HLS INLINE
	stream<ap_uint<SIMDWidth> > in2mvu("StreamingFCLayer_Batch.in2mvu");
	stream<ap_uint<PECount> > mvu2out("StreamingFCLayer_Batch.mvu2out");
	const unsigned int InpPerImage = MatrixW / InStreamW;
	StreamingDataWidthConverter_Batch<InStreamW, SIMDWidth, InpPerImage>(in,
			in2mvu, numReps);
	StreamingMatrixVector_Batch<SIMDWidth, PECount, PopCountWidth, MatrixW,
			MatrixH, WMemCount, TMemCount>(in2mvu, mvu2out, weightMem, thresMem,
			numReps);
	const unsigned int OutPerImage = MatrixH / PECount;
	StreamingDataWidthConverter_Batch<PECount, OutStreamW, OutPerImage>(mvu2out,
			out, numReps);
}

// helper function for fully connected layers with no activation
// instantiates matrix vector unit plus data width converters
template<unsigned int InStreamW, unsigned int OutStreamW,
		unsigned int SIMDWidth, unsigned int PECount,
		unsigned int PopCountWidth, unsigned int MatrixW, unsigned int MatrixH,
		unsigned int WMemCount>
void StreamingFCLayer_NoActivation_Batch(stream<ap_uint<InStreamW> > & in,
		stream<ap_uint<OutStreamW> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const unsigned int numReps) {
#pragma HLS INLINE
	stream<ap_uint<SIMDWidth> > in2mvu("StreamingFCLayer_NoAct_Batch.in2mvu");
	stream<ap_uint<PECount * PopCountWidth> > mvu2out(
			"StreamingFCLayer_NoAct_Batch.mvu2out");
	const unsigned int InpPerImage = MatrixW / InStreamW;
	StreamingDataWidthConverter_Batch<InStreamW, SIMDWidth, InpPerImage>(in,
			in2mvu, numReps);
	StreamingMatrixVector_NoActivation_Batch<SIMDWidth, PECount, PopCountWidth,
			MatrixW, MatrixH, WMemCount>(in2mvu, mvu2out, weightMem, numReps);
	const unsigned int OutPerImage = MatrixH / PECount;
	StreamingDataWidthConverter_Batch<PECount * PopCountWidth, OutStreamW,
			OutPerImage>(mvu2out, out, numReps);
}
