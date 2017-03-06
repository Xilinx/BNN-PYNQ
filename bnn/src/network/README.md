# bnn-pynq

BNN topologies for the PYNQ release, based on the network descriptions reported in the FINN paper as LFC and CNV. It has two scripts in the root folder:
 
 - "make-hw.sh" launches HLS synthesis and the overlay generation for a given configuration.
 
 - "make-sw.sh" generates shared objects that uses the accelerator for the PYNQ for a given configuration. It supports the HW accelerator host code or a SW implementation

This repo also contains one folder per configuration which is structured like this:

 - "<config>/hw" contains the HLS config header file and top-level .cpp that instantiates all the layers and wraps them into an IP
 - "<config>/sw" contains the configuration-specific software code that loads the dataset, initializes memories and launches the accelerator.
