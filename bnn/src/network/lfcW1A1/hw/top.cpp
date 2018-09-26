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
 * @file top.cpp
 *
 * HLS Description of the LFC BNN, with axi-lite based parameter loading (DoMemInit) 
 * and  dataflow architecture of the image inference (DoCompute)
 * 
 *
 *****************************************************************************/

#define CASSERT_DATAFLOW(x)

#include "config.h"
#include "dma.h"
#include "fclayer.h"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"

static BinaryWeights<L0_SIMD,L0_PE,L0_WMEM> weights0;
static BinaryWeights<L1_SIMD,L1_PE,L1_WMEM> weights1;
static BinaryWeights<L2_SIMD,L2_PE,L2_WMEM> weights2;
static BinaryWeights<L3_SIMD,L3_PE,L3_WMEM> weights3;

static ThresholdsActivation<L0_TMEM,L0_PE,L0_API,ap_int<16>,ap_uint<L0_API>> threshs0;
static ThresholdsActivation<L1_TMEM,L1_PE,L1_API,ap_int<16>,ap_uint<L1_API>> threshs1;
static ThresholdsActivation<L2_TMEM,L2_PE,L2_API,ap_int<16>,ap_uint<L2_API>> threshs2;
static ThresholdsActivation<L3_TMEM,L3_PE,L3_API,ap_int<16>,ap_uint<L3_API>> threshs3;

unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0) {
    return in;
  } else {
    return in + padTo - (in % padTo);
  }
}

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val) {
  switch(targetLayer) {
    case 0:
      weights0.m_weights[targetMem][targetInd] = val;
      break;
    case 1:
      threshs0.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 2:
      weights1.m_weights[targetMem][targetInd] = val;
      break;
    case 3:
      threshs1.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 4:
      weights2.m_weights[targetMem][targetInd] = val;
      break;
    case 5:
      threshs2.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 6:
      weights3.m_weights[targetMem][targetInd] = val;
      break;
    case 7:
      threshs3.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
  }
}

void DoCompute(ap_uint<64> *in, ap_uint<64> *out, const unsigned int numReps) {

  hls::stream<ap_uint<64>>     memInStrm("DoCompute.memInStrm");
  hls::stream<ap_uint<L0_PE * (L0_API + L0_APF)>>  inter0("DoCompute.inter0");
  hls::stream<ap_uint<L1_PE * (L1_API + L1_APF)>>  inter1("DoCompute.inter1");
  hls::stream<ap_uint<L2_PE * (L2_API + L2_APF)>>  inter2("DoCompute.inter2");
  hls::stream<ap_uint<64>>     memOutStrm("DoCompute.memOutStrm");

  // TODO: These values are just for comparability and produce the same amount
  //       of FIFO buffer as in the preceding design.
  //       If the resources allow, these depths should eventually go up to
  //       Lx_DEPTH = Lx_MH/Lx_PE for a smooth operation.
  unsigned const  L0_DEPTH = 512 / L0_PE;
  unsigned const  L1_DEPTH = 512 / L1_PE;
  unsigned const  L2_DEPTH = 512 / L2_PE;
  
#pragma HLS DATAFLOW
#pragma HLS stream depth=1024 variable=memInStrm     	// mask memory latency
#pragma HLS stream depth=L0_DEPTH variable=inter0
#pragma HLS stream depth=L1_DEPTH variable=inter1
#pragma HLS stream depth=L2_DEPTH variable=inter2
#pragma HLS stream depth=1024 variable=memOutStrm		// mask memory latency

  const unsigned int inBits = 28 * 28;
  const unsigned int inBitsPadded = 832; // paddedSizeHW(inBits, 64)
  const unsigned int inBytesPadded = inBitsPadded / 8;
  const unsigned int outBits = 64;
  const unsigned int outBitsPadded = 64; // paddedSizeHW(outBits, 64)
  const unsigned int outBytesPadded = outBitsPadded / 8;
  const unsigned int inWordsPerImg = inBitsPadded / 64;
  const unsigned int outWordsPerImg = outBitsPadded / 64;

  Mem2Stream_Batch<64, inBytesPadded>(in, memInStrm, numReps);
  StreamingFCLayer_Batch<L0_MW, L0_MH, L0_SIMD, L0_PE, Recast<XnorMul>>
    (memInStrm, inter0, weights0, threshs0, numReps, ap_resource_lut());
  StreamingFCLayer_Batch<L1_MW, L1_MH, L1_SIMD, L1_PE, Recast<XnorMul>>
    (inter0, inter1, weights1, threshs1, numReps, ap_resource_lut());
  StreamingFCLayer_Batch<L2_MW, L2_MH, L2_SIMD, L2_PE, Recast<XnorMul>>
    (inter1, inter2, weights2, threshs2, numReps, ap_resource_lut());
  StreamingFCLayer_Batch<L3_MW, L3_MH, L3_SIMD, L3_PE, Recast<XnorMul>>
    (inter2, memOutStrm, weights3, threshs3, numReps, ap_resource_lut());
  Stream2Mem_Batch<64, outBytesPadded>(memOutStrm, out, numReps);
}

void BlackBoxJam(ap_uint<64> *in, ap_uint<64> *out, bool doInit,
                 unsigned int targetLayer, unsigned int targetMem,
                 unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val, unsigned int numReps) {
// pragmas for MLBP jam interface
// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=doInit bundle=control
#pragma HLS INTERFACE s_axilite port=targetLayer bundle=control
#pragma HLS INTERFACE s_axilite port=targetMem bundle=control
#pragma HLS INTERFACE s_axilite port=targetThresh bundle=control
#pragma HLS INTERFACE s_axilite port=targetInd bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
// signals to be mapped to the AXI master port (hostmem)
#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem depth=256
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem depth=256
#pragma HLS INTERFACE s_axilite port=out bundle=control

//partition PE arrays
#pragma HLS ARRAY_PARTITION variable=weights0.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights1.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs1.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights2.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs2.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights3.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs3.m_thresholds complete dim=1

  if (doInit) {
    DoMemInit(targetLayer, targetMem, targetInd, targetThresh, val);
  } else {
    DoCompute(in, out, numReps);
  }
}
