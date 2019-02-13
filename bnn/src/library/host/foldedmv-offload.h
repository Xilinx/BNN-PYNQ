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
 * @file foldedmv-offload.h
 *
 * Library of functions for host code and managing HW offload
 * 
 *
 *****************************************************************************/

#pragma once
#include <string>
#include <iostream>
#include "tiny_cnn/tiny_cnn.h"
#include "ap_int.h"

using namespace std;

typedef unsigned long long ExtMemWord;

const unsigned int bitsPerExtMemWord = sizeof(ExtMemWord)*8;

#ifndef VIRTUAL
  #define INPUT_BUF_ENTRIES     3840000
  #define OUTPUT_BUF_ENTRIES    160000
#else
  #define INPUT_BUF_ENTRIES		8192
  #define OUTPUT_BUF_ENTRIES	1024
#endif

#define FOLDEDMV_INPUT_PADCHAR  0

void FoldedMVOffloadBinarized(const ExtMemWord * in, 
                              ExtMemWord * out,
						      const unsigned int inBufWords, 
							  const unsigned int outBufWords, 
							  const unsigned int numImages);

void FoldedMVInit(const char * attachName);

void FoldedMVDeinit();

void FoldedMVLoadLayerMem(std::string dir, 
                          unsigned int peCount, 
						  unsigned int layerNo, 
						  unsigned int linesWMem, 
						  unsigned int linesTMem, 
						  unsigned int numThresh);

void FoldedMVMemSet(unsigned int targetLayer, 
                    unsigned int targetMem, 
					unsigned int targetInd, 
					unsigned int targetThresh, 
					ExtMemWord val);

std::vector<int> testPrebinarized_nolabel_multiple_images(std::vector<tiny_cnn::vec_t> & imgs, 
                                                          const unsigned int labelBits, 
														  float &usecPerImage);

std::vector<int> testPrebinarized_nolabel(std::vector<tiny_cnn::vec_t> & imgs, 
                                          const unsigned int labelBits, 
										  float &usecPerImage);

void testPrebinarized(std::vector<tiny_cnn::vec_t> & imgs, 
                      std::vector<tiny_cnn::label_t> & labels, 
					  const unsigned int labelBits);

void binarizeAndPack(const tiny_cnn::vec_t & in, 
                     ExtMemWord * out, 
					 unsigned int inBufSize=INPUT_BUF_ENTRIES);

void unpackAndDebinarize(const ExtMemWord * in, tiny_cnn::vec_t &out);

unsigned int paddedSize(unsigned int in, unsigned int padTo);

std::string getBNNRoot();

template<typename LowPrecType>
void copyFromLowPrecBuffer(void * buf, tiny_cnn::vec_t & out) {
  LowPrecType * lpbuf = (LowPrecType *) buf;
  for(unsigned int i = 0; i < out.size(); i++) {
    out[i] = (tiny_cnn::float_t) lpbuf[i];
  }
}

template<unsigned int inWidth, unsigned int SIMDWidth>
void quantiseAndPack(const tiny_cnn::vec_t & in, ExtMemWord * out, unsigned int inBufSize=INPUT_BUF_ENTRIES) {
  if((in.size() * inWidth) > (inBufSize * bitsPerExtMemWord)) {
    throw "Not enough space in input buffer";
  }
  // first, fill the target buffer with padding data
  memset(out, 0, inBufSize * sizeof(ExtMemWord));
  ExtMemWord tmpv[bitsPerExtMemWord / inWidth];
  // now pack each quantised value as required.
  for(unsigned int i=0; i < in.size(); i++) {
    ap_fixed<inWidth, 1, AP_RND, AP_SAT> fxdValue = in[i];
    ap_uint<inWidth> uValue = *reinterpret_cast<ap_uint<inWidth> *>(&fxdValue); // Interpret the fixed value as an integer.
    ExtMemWord v = ((ExtMemWord)uValue & (~(ExtMemWord)0 >> (bitsPerExtMemWord - inWidth))); // Zero all bits except for the (bitsPerExtMemWord - inWidth) least significant bits.
    out[i / (bitsPerExtMemWord / inWidth)] |= (v << inWidth*(i % (bitsPerExtMemWord / inWidth)));
  }
}

#if defined(OFFLOAD) && defined(RAWHLS)

#include "bnn-library.h"

void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit, unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val, unsigned int numReps);

extern ExtMemWord * bufIn, * bufOut;

template<typename LowPrecType>
void FoldedMVOffload(const tiny_cnn::vec_t &in, tiny_cnn::vec_t & out, unsigned int offloadID, tiny_cnn::OffloadConvParams * convParams) {
  // binarize input and pack into bit stream
  binarizeAndPack(in, bufIn);

  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)bufIn, (ap_uint<64> *)bufOut, false, 0, 0, 0, 0, 0, 1);

  // unpack output bits and convert output back to float
  if(offloadID == 0xdeadbeef) {
    copyFromLowPrecBuffer<LowPrecType>((void *)bufOut, out);
  } else {
    unpackAndDebinarize(bufOut, out);
  }
}

template<unsigned int inWidth, unsigned int SIMDWidth, typename LowPrecType>
void FixedFoldedMVOffload(const tiny_cnn::vec_t &in, tiny_cnn::vec_t &out, unsigned int offloadID, tiny_cnn::OffloadConvParams * convParams) {
  // binarize input and pack into bit stream
  quantiseAndPack<inWidth, SIMDWidth>(in, bufIn);

  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)bufIn, (ap_uint<64> *)bufOut, false, 0, 0, 0, 0, 0, 1);

  // unpack output bits and convert output back to float
  if(offloadID == 0xdeadbeef) {
    copyFromLowPrecBuffer<LowPrecType>((void *)bufOut, out);
  } else {
    unpackAndDebinarize(bufOut, out);
  }
}


template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
void testPrebuiltCIFAR10(std::vector<tiny_cnn::vec_t> & imgs, std::vector<tiny_cnn::label_t> & labels, const unsigned int numCategories) {
  const unsigned int count = imgs.size();
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // number of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // number of ExtMemWords per output
  const unsigned int pso = 16; //paddedSize(numCategories*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi) {
    throw "Not enough space in accelBufIn";
  }
  if(OUTPUT_BUF_ENTRIES < count*pso) {
    throw "Not enough space in accelBufOut";
  }
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  auto t1 = chrono::high_resolution_clock::now();
  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0,0, count);
  auto t2 = chrono::high_resolution_clock::now();
  // compare against labels
  unsigned int ok = 0, failed = 0;
  tiny_cnn::vec_t outTest(numCategories, 0);
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<LowPrecType>(&packedOut[i * pso], outTest);
    int maxInd = 0;
    LowPrecType maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
    if(maxInd == labels[i]) {
      ok++;
    } else {
      failed++;
    }
  }
  cout << "Succeeded " << ok << " failed " << failed << " accuracy " << 100.0*(float)ok/count << "%" << endl;
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  float usecPerImage = (float)duration / count;
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete[] packedImages;
  delete[] packedOut;
}


template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
std::vector<int>  testPrebuiltCIFAR10_from_image(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int numCategories, float &usecPerImage) {
  const unsigned int count = 1;
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // number of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // number of ExtMemWords per output
  const unsigned int pso = paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi) {
    throw "Not enough space in accelBufIn";
  }
  if(OUTPUT_BUF_ENTRIES < count*pso) {
    throw "Not enough space in accelBufOut";
  }
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;

  auto t1 = chrono::high_resolution_clock::now();
  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, 0, count);
  auto t2 = chrono::high_resolution_clock::now();

  // compare against labels
  unsigned int ok = 0, failed = 0;
  tiny_cnn::vec_t outTest(numCategories, 0);
  copyFromLowPrecBuffer<LowPrecType>(&packedOut[0], outTest);
  std::vector<int> result;
  for(unsigned int j = 0; j < numCategories; j++) {
    result.push_back(outTest[j]);
  }
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete[] packedImages;
  delete[] packedOut;
  return result;
}

template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
std::vector<int> testPrebuiltCIFAR10_multiple_images(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int numCategories, std::vector<int> & detailed_results, float & usecPerImage) {
  const unsigned int count = imgs.size();
  std::vector<int> results;
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // number of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // number of ExtMemWords per output
  const unsigned int pso = paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi)
    throw "Not enough space in accelBufIn";
  if(OUTPUT_BUF_ENTRIES < count*pso)
    throw "Not enough space in accelBufOut";
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  // copy inputs to accelerator
  auto t1 = chrono::high_resolution_clock::now();
  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, 0, count);
  auto t2 = chrono::high_resolution_clock::now();
  // compare against labels
  tiny_cnn::vec_t outTest(numCategories, 0);
  
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<LowPrecType>(&packedOut[i * pso], outTest);
    int maxInd = 0;
    LowPrecType maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
    detailed_results.push_back(outTest[j]);
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
	results.push_back(maxInd);
  }  
  auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete[] packedImages;
  delete[] packedOut;
  return results;
}

#elif defined(OFFLOAD) && !defined(RAWHLS)
#include "platform.hpp"
#include <vector>

extern DonutDriver * thePlatform;
extern void * accelBufIn, * accelBufOut;
extern ExtMemWord * bufIn, * bufOut;

void ExecAccel();

template<typename LowPrecType>
void FoldedMVOffload(const tiny_cnn::vec_t &in, tiny_cnn::vec_t &out, unsigned int offloadID, tiny_cnn::OffloadConvParams * convParams) {
  // always operates on a single image per call for now -- set numImages to 1
  thePlatform->writeJamRegAddr(0x5C, 1);
  // binarize input and pack into bit stream
  binarizeAndPack(in, bufIn);

  // TODO size to pad input to is max(64, PE_SYNGROUP_BITS)
  unsigned int paddedInDim = paddedSize(in.size(), bitsPerExtMemWord);
  // copy into accelerator input
  const unsigned int numInpWords = (paddedInDim / bitsPerExtMemWord);
  thePlatform->copyBufferHostToAccel((void *)bufIn, accelBufIn, sizeof(ExtMemWord) * numInpWords);

  // launch
  ExecAccel();

  if(offloadID == 0xdeadbeef) {
    unsigned int paddedOutDim = paddedSize(out.size() * 16, bitsPerExtMemWord);
    const unsigned int numOutWords = (paddedOutDim / bitsPerExtMemWord);
    thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord) * numOutWords);
    copyFromLowPrecBuffer<LowPrecType>((void *)bufOut, out);
  } else {
    // TODO size to pad input to is max(64, NUM_PE_ELEMENTS)
    unsigned int paddedOutDim = paddedSize(out.size(), bitsPerExtMemWord);

    // copy from accelerator output
    const unsigned int numOutWords = ( paddedOutDim / bitsPerExtMemWord);
    thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord) * numOutWords);

    // unpack output bits and convert output back to float
    unpackAndDebinarize(bufOut, out);
  }
}

template<unsigned int inWidth, unsigned int SIMDWidth, typename LowPrecType>
void FixedFoldedMVOffload(const tiny_cnn::vec_t &in, tiny_cnn::vec_t &out, unsigned int offloadID, tiny_cnn::OffloadConvParams * convParams) {
  // always operates on a single image per call for now -- set numImages to 1
  thePlatform->writeJamRegAddr(0x5C, 1);
  // binarize input and pack into bit stream
  quantiseAndPack<inWidth, SIMDWidth>(in, bufIn);

  // TODO size to pad input to is max(64, PE_SYNGROUP_BITS)
  unsigned int paddedInDim = paddedSize(in.size(), bitsPerExtMemWord);
  // copy into accelerator input
  const unsigned int numInpWords = (paddedInDim / (bitsPerExtMemWord / inWidth));
  thePlatform->copyBufferHostToAccel((void *)bufIn, accelBufIn, sizeof(ExtMemWord) * numInpWords);

  // launch
  ExecAccel();

  if(offloadID == 0xdeadbeef) {
    unsigned int paddedOutDim = paddedSize(out.size() * 16, bitsPerExtMemWord);
    const unsigned int numOutWords = ( paddedOutDim / bitsPerExtMemWord);
    thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord) * numOutWords);
    copyFromLowPrecBuffer<LowPrecType>((void *)bufOut, out);
  } else {
    // TODO size to pad input to is max(64, NUM_PE_ELEMENTS)
    unsigned int paddedOutDim = paddedSize(out.size(), bitsPerExtMemWord);

    // copy from accelerator output
    const unsigned int numOutWords = ( paddedOutDim / bitsPerExtMemWord);
    thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord) * numOutWords);

    // unpack output bits and convert output back to float
    unpackAndDebinarize(bufOut, out);
  }
}

template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
void testPrebuiltCIFAR10(std::vector<tiny_cnn::vec_t> & imgs, std::vector<tiny_cnn::label_t> & labels, const unsigned int numCategories) {
  const unsigned int count = imgs.size();
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // # of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size() * inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // # of ExtMemWords per output
  const unsigned int pso = paddedSize(numCategories * outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi) {
    throw "Not enough space in accelBufIn";
  }
  if(OUTPUT_BUF_ENTRIES < count*pso) {
    throw "Not enough space in accelBufOut";
  }
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  // copy inputs to accelerator
  thePlatform->copyBufferHostToAccel((void *)packedImages, accelBufIn, sizeof(ExtMemWord) * count * psi);
  // set number of images to recognize
  thePlatform->writeJamRegAddr(0x5C, count);
  
  // recognize
  auto t1 = chrono::high_resolution_clock::now();
  ExecAccel();
  auto t2 = chrono::high_resolution_clock::now();
  
  // copy results back to host
  thePlatform->copyBufferAccelToHost(accelBufOut, (void *)packedOut, sizeof(ExtMemWord) * count * pso);
  // compare against labels
  unsigned int ok = 0, failed = 0;
  tiny_cnn::vec_t outTest(numCategories, 0);
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<LowPrecType>(&packedOut[i * pso], outTest);
    int maxInd = 0;
    LowPrecType maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
    if(maxInd == labels[i]) {
      ok++;
    } else {
      failed++;
    }
  }
  cout << "Succeeded " << ok << " failed " << failed << " accuracy " << 100.0 * (float)ok / count << "%" << endl;
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  float usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete[] packedImages;
  delete[] packedOut;
}

template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
std::vector<int> testPrebuiltCIFAR10_from_image(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int numCategories, float &usecPerImage) {
  const unsigned int count = 1;
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // number of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // number of ExtMemWords per output
  const unsigned int pso = paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi) {
    throw "Not enough space in accelBufIn";
  }
  if(OUTPUT_BUF_ENTRIES < count*pso) {
    throw "Not enough space in accelBufOut";
  }
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  // copy inputs to accelerator
  thePlatform->copyBufferHostToAccel((void *)packedImages, accelBufIn, sizeof(ExtMemWord) * count * psi);
  // set number of images to recognize
  thePlatform->writeJamRegAddr(0x5C, count);
  
  // recognize
  auto t1 = chrono::high_resolution_clock::now();
  ExecAccel();
  auto t2 = chrono::high_resolution_clock::now();
  
  // copy results back to host
  thePlatform->copyBufferAccelToHost(accelBufOut, (void *)packedOut, sizeof(ExtMemWord) * count * pso);

  // compare against labels
  unsigned int ok = 0, failed = 0;
  tiny_cnn::vec_t outTest(numCategories, 0);
  copyFromLowPrecBuffer<LowPrecType>(&packedOut[0], outTest);
  std::vector<int> result;
  for(unsigned int j = 0; j < numCategories; j++) {
    result.push_back(outTest[j]);
  }

  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete [] packedImages;
  delete [] packedOut;
  return result;
}

template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
std::vector<int> testPrebuiltCIFAR10_multiple_images(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int numCategories, std::vector<int> & detailed_results, float &usecPerImage) {
  const unsigned int count = imgs.size();
  std::vector<int> results;
  cout << "Packing and interleaving CIFAR-""10 inputs..." << endl;
  // number of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // number of ExtMemWords per output
  const unsigned int pso = paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi) {
    throw "Not enough space in accelBufIn";
  }
  if(OUTPUT_BUF_ENTRIES < count*pso) {
    throw "Not enough space in accelBufOut";
  }
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  // copy inputs to accelerator
  thePlatform->copyBufferHostToAccel((void *)packedImages, accelBufIn, sizeof(ExtMemWord) * count * psi);
  // set number of images to recognize
  thePlatform->writeJamRegAddr(0x5C, count);
  
  // recognize
  auto t1 = chrono::high_resolution_clock::now();
  ExecAccel();
  auto t2 = chrono::high_resolution_clock::now();
  
  // copy results back to host
  thePlatform->copyBufferAccelToHost(accelBufOut, (void *)packedOut, sizeof(ExtMemWord) * count * pso);
  tiny_cnn::vec_t outTest(numCategories, 0);
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<LowPrecType>(&packedOut[i * pso], outTest);
    int maxInd = 0;
    LowPrecType maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
    detailed_results.push_back(outTest[j]);
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
    results.push_back(maxInd);	   	  
  }  

  auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete[] packedImages;
  delete[] packedOut;
  return results;
 }


#endif

