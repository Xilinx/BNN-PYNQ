/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Fully-Connected Layer L0:
 *     MatW =   832 MatH =  1024
 *     SIMD =    64  PE  =    32
 *     WMEM =   416 TMEM =    32
 *     #Ops  = 1703936   Ext Latency  =   416
**/

#define L0_SIMD 64
#define L0_PE 32
#define L0_WMEM 416
#define L0_TMEM 32
#define L0_MW 832
#define L0_MH 1024
#define L0_WPI 1
#define L0_API 2
#define L0_WPF 0
#define L0_APF 0

/**
 * Fully-Connected Layer L1:
 *     MatW =  1024 MatH =  1024
 *     SIMD =    32  PE  =    64
 *     WMEM =   512 TMEM =    16
 *     #Ops  = 2097152   Ext Latency  =   512
**/

#define L1_SIMD 32
#define L1_PE 64
#define L1_WMEM 512
#define L1_TMEM 16
#define L1_MW 1024
#define L1_MH 1024
#define L1_WPI 1
#define L1_API 2
#define L1_WPF 0
#define L1_APF 0

/**
 * Fully-Connected Layer L2:
 *     MatW =  1024 MatH =  1024
 *     SIMD =    64  PE  =    32
 *     WMEM =   512 TMEM =    32
 *     #Ops  = 2097152   Ext Latency  =   512
**/

#define L2_SIMD 64
#define L2_PE 32
#define L2_WMEM 512
#define L2_TMEM 32
#define L2_MW 1024
#define L2_MH 1024
#define L2_WPI 1
#define L2_API 2
#define L2_WPF 0
#define L2_APF 0

/**
 * Fully-Connected Layer L3:
 *     MatW =  1024 MatH =    64
 *     SIMD =     8  PE  =    16
 *     WMEM =   512 TMEM =     4
 *     #Ops  = 131072   Ext Latency  =   512
**/

#define L3_SIMD 8
#define L3_PE 16
#define L3_WMEM 512
#define L3_TMEM 4
#define L3_MW 1024
#define L3_MH 64
#define L3_WPI 1
#define L3_API 1
#define L3_WPF 0
#define L3_APF 0

#endif //__LAYER_CONFIG_H_
