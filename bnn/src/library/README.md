# bnn-library

This is the FINN library and contains four folders:

- "hls" contains the FINN hardware library (the templatized streaming components in HLS).

- "host" contains C++ host code to load weights into and launch the accelerator(s). It still depends on tiny-cnn to load the test images, and on the HLS config header file to know how many layers there are. 

- "script" contains a set of Vivado scripts to generate the block design and bitstream generation of the HW accelerators. Those scripts are used when executing make-hw.

- "driver" contains a set of source files to handle the communication with the HW accelerator.

