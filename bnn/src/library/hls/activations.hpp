/******************************************************************************
 *  Copyright (c) 2017, Xilinx, Inc.
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
 *******************************************************************************/

/*******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file activations.hpp
 *
 *  Library of templated HLS classes for BNN deployment. 
 *  This file lists a set of classes used to implement  
 *  threshold memory in neural network. 
 *
 *  This project has received funding from the European Union's Framework
 *  Programme for Research and Innovation Horizon 2020 (2014-2020) under
 *  the Marie Skłodowska-Curie Grant Agreement No. 751339.
 *
 *******************************************************************************/

#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

/**
 * General contract for activation functions.
 *
 * This class itself has no formal significance for the implementation
 * of the MVAU. Implementations of activation functions are encouraged
 * to implement it nonetheless to guarantee appropriate function
 * signatures.
 */
template<typename TA, typename TO>
class Activation {
public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

  /**
   * Compute the activation of the passed accumulator value accu in row idx.
   */
  TO activate(unsigned const  nf, unsigned const  pe, TA const &accu) const;
};

/**
 * A no-op activation that simply outputs the computed accumulator
 * output as the final result.
 */
template<typename T>
class PassThroughActivation : public Activation<T, T> {
public:
  T activate(unsigned const  nf, unsigned const  pe, T const &accu) const {
#pragma HLS inline
    return  accu;
  }
};

/**
 * Use a simple global threshold comparison as activation function.
 *
 * The constant threshold is initialized at construction.
 * The default comparison returns true if the threshold value is
 * smaller than the passed accumulator value.
 */
template<typename TA, typename Compare = std::less<TA>>
class ThresholdActivation : public Activation<TA, bool> {
  TA const  m_threshold;
public:
  ThresholdActivation(TA const &threshold) : m_threshold(threshold) {
#pragma HLS inline
  }

public:
  bool activate(unsigned const  nf, unsigned const  pe, TA const &accu) const {
#pragma HLS inline
    return  Compare()(m_threshold, accu);
  }
};

/**
 * Use a simple per-row threshold comparison as activation function.
 *
 * The thresholds are taken from an array indexed by output row.
 * It is currently public to allow direct initialization and
 * to make its name accessible for top-level HLS pragmas.
 *
 * The default comparison returns true if the threshold value defined for
 * the indexed row is smaller than the passed accumulator value.
 */
template<unsigned NF, unsigned PE, unsigned NumTH, 
	 typename TA, typename TR, int ActVal = 0, typename Compare = std::less<TA>>
class ThresholdsActivation {
public:
  TA m_thresholds[PE][NF][NumTH];
  
public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

public:
  TR activate(unsigned const  nf, unsigned const  pe,  TA const &accu) const {
#pragma HLS inline
    TR result=ActVal;
	for(unsigned int i=0; i< NumTH; i++){
#pragma HLS unroll
      result+=Compare()(m_thresholds[pe][nf][i], accu);
    }
    return result;
  }
};

#endif
