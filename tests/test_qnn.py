#   Copyright (c) 2016, Xilinx, Inc.
#   All rights reserved.
# 
#   Redistribution and use in source and binary forms, with or without 
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright 
#       notice, this list of conditions and the following disclaimer in the 
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its 
#       contributors may be used to endorse or promote products derived from 
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from pynq import Xlnk
from PIL import Image
import bnn

# Testing LFC with MNIST dataset - HW
def test_mnist():
	# load test image
	BNN_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
	test_image_mnist   = os.path.join(BNN_ROOT_DIR, 'Test_image', '3.image-idx3-ubyte')

	# Testing Hardware

	# Testing LFC-W1A1
	classifier = bnn.LfcClassifier(bnn.NETWORK_LFCW1A1,"mnist",bnn.RUNTIME_HW)
	out = classifier.classify_mnist(test_image_mnist)
	print("Inferred class: ", out)

	assert out==3, \
		'MNIST HW test failed for LFCW1A1'

	# Testing LFC-W1A2
	classifier = bnn.LfcClassifier(bnn.NETWORK_LFCW1A2,"mnist",bnn.RUNTIME_HW)
	out = classifier.classify_mnist(test_image_mnist)
	print("Inferred class: ", out)

	assert out==3, \
		'MNIST HW test failed for LFCW1A2'

	# Testing Software

	# Testing LFC-W1A1
	w1a1 = bnn.LfcClassifier(bnn.NETWORK_LFCW1A1,"mnist",bnn.RUNTIME_SW)
	out = w1a1.classify_mnist(test_image_mnist)	
	print("Inferred class: ", out)
	assert out==3, \
		'MNIST SW test failed for LFC W1A1'

	# Testing LFC-W1A2
	w1a2 = bnn.LfcClassifier(bnn.NETWORK_LFCW1A2,"mnist",bnn.RUNTIME_SW)
	out = w1a2.classify_mnist(test_image_mnist)	
	print("Inferred class: ", out)
	assert out==3, \
		'MNIST SW test failed for LFC W1A2'

	print("test finished with no errors!")
	xlnk = Xlnk()
	xlnk.xlnk_reset()

def test_cifar10():
	# load test image
	BNN_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
	test_image_cifar10 = os.path.join(BNN_ROOT_DIR, 'Test_image', 'deer.jpg')
	test_image_cifar10 = Image.open(test_image_cifar10)

	# Testing Hardware
	# Testing CNV-W1A1	
	classifier = bnn.CnvClassifier(bnn.NETWORK_CNVW1A1,"cifar10", bnn.RUNTIME_HW)
	out=classifier.classify_image(test_image_cifar10)
	print("Inferred class: ", out)

	assert out==4, \
		'Cifar10 HW test failed for CNV-W1A1'

	# Testing CNV-W1A2	
	classifier = bnn.CnvClassifier(bnn.NETWORK_CNVW1A2,"cifar10", bnn.RUNTIME_HW)
	out=classifier.classify_image(test_image_cifar10)
	print("Inferred class: ", out)

	assert out==4, \
		'Cifar10 HW test failed for CNV-W1A2'

	# Testing CNV-W2A2	
	classifier = bnn.CnvClassifier(bnn.NETWORK_CNVW2A2,"cifar10", bnn.RUNTIME_HW)
	out=classifier.classify_image(test_image_cifar10)
	print("Inferred class: ", out)

	assert out==4, \
		'Cifar10 HW test failed for CNV-W2A2'

	# Testing Software

	# Testing CNV-W1A1	
	w1a1 = bnn.CnvClassifier(bnn.NETWORK_CNVW1A1,"cifar10",bnn.RUNTIME_SW)
	out=w1a1.classify_image(test_image_cifar10)
	print("Inferred class: ", out)

	assert out==4, \
		'Cifar10 SW test failed for CNV-W1A1'

	# Testing CNV-W1A2	
	w1a2 = bnn.CnvClassifier(bnn.NETWORK_CNVW1A2,"cifar10",bnn.RUNTIME_SW)
	out=w1a2.classify_image(test_image_cifar10)
	print("Inferred class: ", out)

	assert out==4, \
		'Cifar10 SW test failed for CNV-W1A2'

	# Testing CNV-W2A2	
	w2a2 = bnn.CnvClassifier(bnn.NETWORK_CNVW2A2,"cifar10",bnn.RUNTIME_SW)
	out=w2a2.classify_image(test_image_cifar10)
	print("Inferred class: ", out)

	assert out==4, \
		'Cifar10 SW test failed for CNV-W2A2'

	print("test finished with no errors!")
	xlnk = Xlnk();
	xlnk.xlnk_reset()

def test_svhn():
	# load test image
	BNN_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
	test_image_svhn = os.path.join(BNN_ROOT_DIR, 'Test_image', '6.png')
	test_image_svhn = Image.open(test_image_svhn)

	# Testing Hardware
	# Only testing CNV-W1A1	as parameters are only available to this precision
	classifier = bnn.CnvClassifier(bnn.NETWORK_CNVW1A1,"streetview", bnn.RUNTIME_HW)
	sw_classifier = bnn.CnvClassifier(bnn.NETWORK_CNVW1A1,"streetview", bnn.RUNTIME_SW)
	out=classifier.classify_image(test_image_svhn)
	print("Inferred class: ", out)

	assert out== 5, \
		'SVHN HW test failed for CNV-W1A1'

	#Testing Software
	out=sw_classifier.classify_image(test_image_svhn)
	print("Inferred class: ", out)

	assert out== 5, \
		'SVHN SW test failed for CNV-W1A1'

	print("test finished with no errors!")
	xlnk = Xlnk();
	xlnk.xlnk_reset()


def test_road_sign():
	# load test image
	BNN_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
	test_image_road = os.path.join(BNN_ROOT_DIR, 'Test_image', 'stop.jpg')
	test_image_road = Image.open(test_image_road)

	# Testing Hardware
	# Only testing CNV-W1A1	as parameters are only available to this precision
	classifier = bnn.CnvClassifier(bnn.NETWORK_CNVW1A1,"road-signs", bnn.RUNTIME_HW)
	sw_classifier = bnn.CnvClassifier(bnn.NETWORK_CNVW1A1,"road-signs", bnn.RUNTIME_SW)
	out=classifier.classify_image(test_image_road)
	print("Inferred class: ", out)

	assert out==14, \
		'Road sign HW test failed for CNV-W1A1'

	# Testing Software
	out = sw_classifier.classify_image(test_image_road)
	print("Inferred class: ", out)

	assert out== 14, \
		'Road sign SW test failed for CNV-W1A1'

	print("test finished with no errors!")
	xlnk = Xlnk();
	xlnk.xlnk_reset()
