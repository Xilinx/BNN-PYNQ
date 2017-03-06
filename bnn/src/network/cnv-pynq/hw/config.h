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
 * @file config.h
 *
 * Description of the topology parameters of the CNV BNN, and folding factors 
 * of the hardware implemetation (PE-SIMD) as described in FINN paper
 * 
 *
 *****************************************************************************/

// layer 0 (conv)
// layer config
/* 
Using peCount = 16 simdCount = 3 for engine 0
Extracting conv-BN complex, OFM=64 IFM=3 k=3
Layer 0: 64 x 27
WMem = 36 TMem = 4
*/
#define L0_K  3
#define L0_IFM_CH  3
#define L0_IFM_DIM  32
#define L0_OFM_CH 64
#define L0_OFM_DIM  30
// hardware config
#define L0_SIMD	3
#define L0_PE	16
#define L0_WMEM	36
#define L0_TMEM	4
// layer 1 (conv)
// layer config
/* 
Using peCount = 32 simdCount = 32 for engine 1
Extracting conv-BN complex, OFM=64 IFM=64 k=3
Layer 1: 64 x 576
WMem = 36 TMem = 2
*/
#define L1_K  3
#define L1_IFM_CH  64
#define L1_IFM_DIM  30
#define L1_OFM_CH 64
#define L1_OFM_DIM 28
// hardware config
#define L1_SIMD	32
#define L1_PE	32
#define L1_WMEM	36
#define L1_TMEM	2
// layer 2 (conv)
// layer config
/* 
Using peCount = 16 simdCount = 32 for engine 2
Extracting conv-BN complex, OFM=128 IFM=64 k=3
Layer 2: 128 x 576
WMem = 144 TMem = 8
*/
#define L2_K  3
#define L2_IFM_CH  64
#define L2_IFM_DIM  14
#define L2_OFM_CH 128
#define L2_OFM_DIM 12
// hardware config
#define L2_SIMD	32
#define L2_PE	16
#define L2_WMEM	144
#define L2_TMEM	8
// layer 3 (conv)
// layer config
/* 
Using peCount = 16 simdCount = 32 for engine 3
Extracting conv-BN complex, OFM=128 IFM=128 k=3
Layer 3: 128 x 1152
WMem = 288 TMem = 8
*/
#define L3_K  3
#define L3_IFM_CH  128
#define L3_IFM_DIM  12
#define L3_OFM_CH 128
#define L3_OFM_DIM 10
// hardware config
#define L3_SIMD	32
#define L3_PE	16
#define L3_WMEM	288
#define L3_TMEM	8
// layer 4 (conv)
// layer config
/* 
Using peCount = 4 simdCount = 32 for engine 4
Extracting conv-BN complex, OFM=256 IFM=128 k=3
Layer 4: 256 x 1152
WMem = 2304 TMem = 64
*/
#define L4_K  3
#define L4_IFM_CH  128
#define L4_IFM_DIM  5
#define L4_OFM_CH 256
#define L4_OFM_DIM 3
// hardware config
#define L4_SIMD	32
#define L4_PE	4
#define L4_WMEM	2304
#define L4_TMEM 64
// layer 5 (conv)
// layer config
/* 
Using peCount = 1 simdCount = 32 for engine 5
Extracting conv-BN complex, OFM=256 IFM=256 k=3
Layer 5: 256 x 2304
WMem = 18432 TMem = 256
*/
#define L5_K  3
#define L5_IFM_CH  256
#define L5_IFM_DIM  3
#define L5_OFM_CH 256
#define L5_OFM_DIM 1
// hardware config
#define L5_SIMD	32
#define L5_PE	1
#define L5_WMEM	18432
#define L5_TMEM	256
// layer 6 (fc)
/* 
Using peCount = 1 simdCount = 4 for engine 6
Extracting FCBN complex, ins = 256 outs = 512
Interleaving 256 channels in fully connected layer...
Layer 6: 512 x 256
WMem = 32768 TMem = 512
*/
#define L6_SIMD	4
#define L6_PE	1
#define L6_MH 512
#define L6_MW 256
#define L6_WMEM	32768
#define L6_TMEM	512
// layer 7 (fc)
/* 
Using peCount = 1 simdCount = 8 for engine 7
Extracting FCBN complex, ins = 512 outs = 512
Layer 7: 512 x 512
WMem = 32768 TMem = 512
*/
#define L7_SIMD	8
#define L7_PE	1
#define L7_MH 512
#define L7_MW 512
#define L7_WMEM	32768
#define L7_TMEM	512
// layer 8 (fc, no activation so no threshold memory)
/*
Using peCount = 4 simdCount = 1 for engine 8
Extracting FCBN complex, ins = 512 outs = 10
Layer 8: 12 x 512
WMem = 1536 TMem = 3
 */
#define L8_SIMD 1
#define L8_PE 4
#define L8_MH 64
#define L8_MW 512
#define L8_WMEM 8192
#define L8_TMEM 16
