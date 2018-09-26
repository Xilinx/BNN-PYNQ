# bnn-pynq

Network topologies for the PYNQ and Ultra96 release, based on the network descriptions reported in the FINN paper as LFC and CNV. Now, different precision are available within each topology (W1A1 and W1A2 for both and additional W2A2 for CNV). Two scripts are located in the root folder:
 
 - "make-hw.sh" launches HLS synthesis and the overlay generation for a given configuration.
 
 - "make-sw.sh" generates shared objects that uses the accelerator for the PYNQ or Ultra96 for a given configuration. It supports the HW accelerator host code or a SW implementation and automatically detects the Board.

This repo also contains one folder per configuration which is structured like this:

 - "<network config>/hw" contains the HLS config header file and top-level .cpp that instantiates all the layers and wraps them into an IP
 - "<network config>/sw" contains the configuration-specific software code that loads the dataset, initializes memories and launches the accelerator.
 