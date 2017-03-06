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
 * Description of the topology parameters of the LFC BNN, and folding factors 
 * of the hardware implemetation (PE-SIMD) as described in FINN paper
 * 
 *
 *****************************************************************************/

/*Extracting FCBN complex, ins = 784 outs = 1024
Layer 0: 1024 x 832, SIMD = 64, PE = 32
WMem = 416 TMem = 32
Extracting FCBN complex, ins = 1024 outs = 1024
Layer 1: 1024 x 1024, SIMD = 32, PE = 64
WMem = 512 TMem = 16
Warning: Zero or negative (val=-116) threshold detected.
Warning: Zero or negative (val=-159) threshold detected.
Warning: Zero or negative (val=-10272) threshold detected.
Extracting FCBN complex, ins = 1024 outs = 1024
Layer 2: 1024 x 1024, SIMD = 64, PE = 32
WMem = 512 TMem = 32
Extracting FCBN complex, ins = 1024 outs = 10
Layer 3: 16 x 1024, SIMD = 8, PE = 16
WMem = 128 TMem = 1
Config header file:*/



#define L0_SIMD 64
#define L0_PE 32
#define L0_WMEM 416
#define L0_TMEM 32
#define L0_MW 832
#define L0_MH 1024

#define L1_SIMD 32
#define L1_PE 64
#define L1_WMEM 512
#define L1_TMEM 16
#define L1_MW 1024
#define L1_MH 1024

#define L2_SIMD 64
#define L2_PE 32
#define L2_WMEM 512
#define L2_TMEM 32
#define L2_MW 1024
#define L2_MH 1024

#define L3_SIMD 8
#define L3_PE 16
#define L3_WMEM 512
#define L3_TMEM 4
#define L3_MW 1024
#define L3_MH 64

