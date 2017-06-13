#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))

// sliding window unit that produces several vectors simultaneously for feeding
// a matrix multiple vectors unit

template<unsigned int ConvKernelDim, 
		unsigned int IFMChannels, 
		unsigned int IFMDim, 
		unsigned int OFMDim,
		unsigned int InpWidth=1, // bit per pixel
		unsigned int Stride = 1>
void StreamingConvolutionInputGenerator_Batch(
		stream<ap_uint<IFMChannels*InpWidth> > & in,
		stream<ap_uint<IFMChannels*InpWidth> > & out,
		const unsigned int numReps) {
			
	constexpr unsigned int number_blocks = ConvKernelDim/Stride + 1 ;
  ap_uint<IFMChannels*InpWidth> inputBuf[number_blocks][Stride * IFMDim];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
#pragma HLS RESOURCE variable inputBuf core=RAM_2P
	constexpr unsigned int cycles_write_block = (OFMDim * ConvKernelDim * ConvKernelDim);
	constexpr unsigned int cycles_read_block = Stride * IFMDim;
	constexpr unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
	const unsigned int baseIter = IFMDim * ConvKernelDim // Initial buffer
			+ OFMDim * MAX(cycles_write_block,cycles_read_block);
	unsigned int counter_internal_block = 0;
	unsigned int current_block_write = 0;
	unsigned int next_block_write = 0;	
	unsigned int current_line = 0;
	unsigned int read_block = 0; 
	unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0;
#pragma HLS reset variable=inp
	for (unsigned int count_image = 0; count_image < numReps; count_image++) {
		for (unsigned int i = 0; i < baseIter; i++) {
	#pragma HLS PIPELINE II=1
			if (inp < IFMDim * ConvKernelDim) // Initial buffer
				{
				ap_uint<IFMChannels*InpWidth> inElem;
				inElem = in.read();
				inputBuf[current_block_write][current_line] = inElem;
				current_line++;
				inp++;
				if (current_line == Stride * IFMDim)
					{
					current_line = 0;
					current_block_write++;
					if (current_block_write == number_blocks)
						current_block_write=0;
					read_block++;
					counter_internal_block = 0;
					}
				}
			else
				{
				if (counter_internal_block < cycles_write_block-1) // We are writing output
				{
					unsigned int current_block_read = (current_block_write + 1 + k_y / Stride);
					if (current_block_read >= number_blocks)
						current_block_read-= number_blocks;
					unsigned int current_line_in_block = (k_y%Stride) * IFMDim + ofm_x*Stride + k_x;
					ap_uint<IFMChannels*InpWidth> outElem = inputBuf[current_block_read][current_line_in_block];
					out.write(outElem);
					k_x++;
					if (k_x == ConvKernelDim) {
						k_x = 0;
						k_y++;
						if (k_y == ConvKernelDim) {
							k_y = 0;
							ofm_x ++;
							if (ofm_x == OFMDim) {
								ofm_x = 0;
								ofm_y++;
								if (ofm_y == OFMDim) {
									ofm_y = 0;
									inp = 0;
								}
							}
						}
					}
				}
				if ((counter_internal_block < cycles_read_block-1) && (read_block<IFMDim/Stride)) // In parallel we write in the buffer, in the current block write if we still need to
				{
					ap_uint<IFMChannels*InpWidth> inElem;
					inElem = in.read();
					inputBuf[current_block_write][current_line] = inElem;
#pragma AP dependence variable=inputBuf intra false
					current_line++;
					if (current_line == Stride * IFMDim) // We read the whole block, we change the next block in which we want to we
					{
						current_line = 0;
						read_block++;
						current_block_write++;
						if (current_block_write == number_blocks)
							current_block_write=0;
#pragma AP dependence variable=current_block_write intra false	
					}
				}
				counter_internal_block++;
				if (counter_internal_block == (max_cycles-1))
				{
					counter_internal_block = 0;
				}
			}
		} // End base_iter
	read_block = 0;
	} // End count_image
} // End generator
