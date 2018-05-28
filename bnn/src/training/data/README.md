# Datasets

For the Cifar10 and MNIST networks, [pylearn2](http://deeplearning.net/software/pylearn2/) to is used to provide the training data.
For the GTSRB example, we need to download the GTSRB dataset from [http://benchmark.ini.rub.de/](http://benchmark.ini.rub.de/).
Furthermore, for the GTSRB example, we also provide a \"junk\" class,
which is used to avoid false positives during inference, for non-traffic sign related images.

## German Traffic Sign Detection Benchmark Junk Class

We also provide an extra set of images to be used as an extra output in the network.
As stated above, this is to avoid false positives during inference.

This dataset, comprises of images from GTSDB which has been broken into small patches.
A subset of these patch were then chosen which had:
* no traffic signs in them; and
* as much variation as possible (e.g., trees, road, sky and cars).

The variation in the chosen patches was done to ensure the inference models were
as resilient as possible to classifying non-traffic sign related images.

## Acknowledgements

Thanks to the [Institut F&#252;r Neuroinformatik Benchmark Team](http://benchmark.ini.rub.de/)
for allowing us to provide you with this modified version of the GTSDB to provide background
images.

If you plan to use this work, please cite the [GTSDB](http://benchmark.ini.rub.de/?section=gtsdb&subsection=news)
paper, and explain the modifications stated above.
The GTSDB citation is provided below:

```
@inproceedings{Houben-IJCNN-2013,
   author = {Sebastian Houben and Johannes Stallkamp and Jan Salmen and Marc Schlipsing and Christian Igel},
   booktitle = {International Joint Conference on Neural Networks},
   title = {Detection of Traffic Signs in Real-World Images: The {G}erman {T}raffic {S}ign {D}etection {B}enchmark},
   number = {1288},
   year = {2013},
}
```
