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
 *****************************************************************************
 *  Authors: Thomas B. Preußer <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *
 *  This project has received funding from the European Union's Framework
 *  Programme for Research and Innovation Horizon 2020 (2014-2020) under
 *  the Marie Skłodowska-Curie Grant Agreement No. 751339.
 */
#ifndef WEIGHTS_HPP
#define WEIGHTS_HPP

#include <ap_int.h>

/**
 * A binary weight storage adapter that translates the internal
 * organization optimized for storage to the generalized access
 * by the MVAU.
 */
template<unsigned SIMD, unsigned PE, unsigned TILES>
class BinaryWeights {
public:
  ap_uint<SIMD>  m_weights[PE][TILES];

private:
  /**
   * Temporary container for the tile index to implement the
   * memory access in pe -> tile order.
   */
  class TileIndex {
    BinaryWeights const &m_par;
    unsigned      const  m_idx;

  public:
    TileIndex(BinaryWeights const &par, unsigned const  idx)
      : m_par(par), m_idx(idx) {
#pragma HLS inline
    }

  public:
    ap_uint<SIMD> operator[](unsigned const  pe) const {
#pragma HLS inline
      return  m_par.m_weights[pe][m_idx];
    }
  };

public:
  TileIndex weights(unsigned const  tile) const {
#pragma HLS inline
    return  TileIndex(*this, tile);
  }
};
#endif
