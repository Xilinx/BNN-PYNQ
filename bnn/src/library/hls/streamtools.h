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
 * @file stream-tools.h
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file lists a set of convenience funtions used to adapt stream size, 
 * remove unnecessary streams (padding) and casting
 * 
 *
 *****************************************************************************/

// only let the first X elements of a stream to pass through, the remainder
// are consumed from input but not re-emitted from the output
// useful for getting rid of e.g. padding words
template<unsigned int DataWidth,		// stream width
		unsigned int NumAllowed, 	// number of words to pass through
		unsigned int NumTotal // total number of words (NumTotal-NumAllowed swallowed)
>
void StreamLimiter(stream<ap_uint<DataWidth> > & in,
		stream<ap_uint<DataWidth> > & out) {
	CASSERT_DATAFLOW(NumTotal >= NumAllowed);
	unsigned int numLeft = NumAllowed;
	for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<DataWidth> e = in.read();
		if (numLeft > 0) {
			out.write(e);
			numLeft--;
		}
	}
}

template<unsigned int DataWidth,		// stream width
		unsigned int NumAllowed, 	// number of words to pass through
		unsigned int NumTotal // total number of words (NumTotal-NumAllowed swallowed)
>
void StreamLimiter_Batch(stream<ap_uint<DataWidth> > & in,
		stream<ap_uint<DataWidth> > & out, unsigned int numReps) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		StreamLimiter<DataWidth, NumAllowed, NumTotal>(in, out);
	}
}

template<typename InT, typename OutT>
void StreamingCast(stream<InT> & in, stream<OutT> & out, unsigned int numReps) {
  for(unsigned int i = 0; i < numReps; i++) {
#pragma HLS PIPELINE II=1
    out.write((OutT) in.read());
  }
}

template<unsigned int InWidth,		// width of input stream
		unsigned int OutWidth,		// width of output stream
		unsigned int NumInWords		// number of input words to process
>
void StreamingDataWidthConverter_Batch(stream<ap_uint<InWidth> > & in,
		stream<ap_uint<OutWidth> > & out, const unsigned int numReps) {
	if (InWidth > OutWidth) {
		// emit multiple output words per input word read
		CASSERT_DATAFLOW(InWidth % OutWidth == 0);
		const unsigned int outPerIn = InWidth / OutWidth;
		const unsigned int totalIters = NumInWords * outPerIn * numReps;
		unsigned int o = 0;
		ap_uint<InWidth> ei = 0;
		for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
			// read new input word if current out count is zero
			if (o == 0)
				ei = in.read();
			// pick output word from the rightmost position
			ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
			out.write(eo);
			// shift input to get new output word for next iteration
			ei = ei >> OutWidth;
			// increment written output count
			o++;
			// wraparound indices to recreate the nested loop structure
			if (o == outPerIn) {
				o = 0;
			}
		}
	} else if (InWidth == OutWidth) {
		// straight-through copy
		for (unsigned int i = 0; i < NumInWords * numReps; i++) {
#pragma HLS PIPELINE II=1
			ap_uint<InWidth> e = in.read();
			out.write(e);
		}

	} else { // InWidth < OutWidth
		// read multiple input words per output word emitted
		CASSERT_DATAFLOW(OutWidth % InWidth == 0);
		const unsigned int inPerOut = OutWidth / InWidth;
		const unsigned int totalIters = NumInWords * numReps;
		unsigned int i = 0;
		ap_uint<OutWidth> eo = 0;
		for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
			// read input and shift into output buffer
			ap_uint<InWidth> ei = in.read();
			eo = eo >> InWidth;
			eo(OutWidth - 1, OutWidth - InWidth) = ei;
			// increment read input count
			i++;
			// wraparound logic to recreate nested loop functionality
			if (i == inPerOut) {
				i = 0;
				out.write(eo);
			}
		}
	}
}

template<unsigned IW, unsigned OW, unsigned N>
 class WidthAdjustedInputStream {
  hls::stream<ap_uint<OW>>  m_target;

 public:
  WidthAdjustedInputStream(hls::stream<ap_uint<IW> >&  source, unsigned const  reps) {
    StreamingDataWidthConverter_Batch<IW, OW, N>(source, m_target, reps);
  }
  ~WidthAdjustedInputStream() {}

 public:
  operator hls::stream<ap_uint<OW> >&() {
    return  m_target;
  }
};
template<unsigned W, unsigned N>
 class WidthAdjustedInputStream<W, W, N> {

  hls::stream<ap_uint<W>> &m_source;

 public:
  WidthAdjustedInputStream(hls::stream<ap_uint<W> >&  source, unsigned const  reps) : m_source(source) {}
  ~WidthAdjustedInputStream() {}

 public:
  operator hls::stream<ap_uint<W> >&() {
    return  m_source;
  }
};


template<unsigned IW, unsigned OW, unsigned N>
 class WidthAdjustedOutputStream {
  hls::stream<ap_uint<IW>>  m_buffer;
  hls::stream<ap_uint<OW>> &m_target;
  unsigned const  m_reps;

 public:
  WidthAdjustedOutputStream(hls::stream<ap_uint<OW> >&  target, unsigned const  reps)
    : m_target(target), m_reps(reps) {}
  ~WidthAdjustedOutputStream() {
    StreamingDataWidthConverter_Batch<IW, OW, N>(m_buffer, m_target, m_reps);
  }

 public:
  operator hls::stream<ap_uint<IW> >&() {
    return  m_buffer;
  }
};
template<unsigned W, unsigned N>
 class WidthAdjustedOutputStream<W, W, N> {
  hls::stream<ap_uint<W>> &m_target;

 public:
  WidthAdjustedOutputStream(hls::stream<ap_uint<W> >&  target, unsigned const  reps)
    : m_target(target) {}
  ~WidthAdjustedOutputStream() {}

 public:
  operator hls::stream<ap_uint<W> >&() {
    return  m_target;
  }
};
