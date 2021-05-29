# PHP FANN wrapper

This is a PHP wrapper for [FANN (Fast Artificial Neural Network) library](http://leenissen.dk/fann/wp/).

## API

The API is documented on http://www.php.net/manual/en/book.fann.php where is the complete documentation for PHP FANN.

The API is very similar to the official [FANN C API](http://leenissen.dk/fann/html/files/fann-h.html). Just functions for fixed `fann_type` have not been mapped because PHP always support `float`. In addition unnecessary arguments for some functions have been left out (for example array length that is not necessary for PHP arrays).

## Installation

The extension can be installed on Linux and Windows.

### Linux

Before you start installation make sure that `libfann` is installed on your system. It's part of the main repository in the most Linux distributions (search for `fann`). If not you need to install it first. Either download it from the [official site](http://leenissen.dk/fann/wp/) or get it from your distro repository. For example on Ubuntu:
```
$ sudo apt-get install libfann-dev
```
Fann installation can be skipped if an RPM for Fedora is used (`libfann` is in the package dependencies).

If the library is re-installed manually, then all old library file should be removed before re-installing otherwise the old library version could be linked.

#### Fedora

The RPM package for PHP FANN is available in Remi's repository: http://rpms.famillecollet.com/

It is available for Fedora, RHEL and clones (CentOS, SC and others).

After downloading remi-release RPM, the package can be installed by executing following command:
```
$ sudo yum --enablerepo=remi install php-pecl-fann
```

#### PECL

This extension is available on PECL. The installation is very simple. Just run:

```
$ sudo pecl install fann
```

#### Manual Installation

It's important to have a git installed as it's necessary for recursive fetch of
[phpc](https://github.com/bukka/phpc).

First clone recursively the repository
```
git clone --recursive https://github.com/bukka/php-fann.git
```

Then go to the created source directory and compile the extension. You need to have a php development package installed (command `phpize` must be available).
```
cd php-fann
phpize
./configure --with-fann
make
sudo make install
```

If you are rebuilding the extension and see warning about Libtool version mismatch error, try to run `phpize --clean` or if it doesn't help, try
```
aclocal && libtoolize --force && autoreconf
```
and then run the compilation steps starting with `phpize` again.

Finally you need to add
```
extension=fann.so
```
to the php.ini

### Windows

Precompiled binary `dll` libraries for php-fann and libfann are available on [the PECL fann page](http://pecl.php.net/package/fann). The compiled version of libfann is 2.2.

## Examples

There are four example projects: [Logic Gates](examples/logic_gates/), [Scaling Data](examples/ScalingData/), [OCR](examples/ocr/) & [Pathfinder](examples/pathfinder/).

#### Logic Gates

###### Simple

The Simple example trains a single neural network to perform the XOR operation.

[simple_train.php](examples/logic_gates/simple_train.php)

[simple_train_epoch.php](examples/logic_gates/simple_train_epoch.php)

[simple_test.php](examples/logic_gates/simple_test.php)

[simple_merge.php](examples/logic_gates/simple_test.php) 


###### All

The All example trains 7 seperate neural networks to perform the AND, NAND, NOR, NOT, OR, XNOR & XOR operations.

[train_all.php](examples/logic_gates/train_all.php)

[test_all.php](examples/logic_gates/test_all.php)



#### Scaling Data

This example uses the XOR dataset with negative one represented as zero and one represented as one-hundred and demonstrates how to scale those values so that FANN can understand them and then how to de-scale the value FANN returns so that you can understand it.

* Scaling allows you to take raw data numbers like -1234.975 or 4502012 in your dataset and convert them into an input/output range that your neural network can understand. 

* De-scaling lets you take the scaled data and convert it back into the original range.

[ScaledXOR.php](examples/ScalingData/ScaledXOR.php)

[scale_test.data](examples/ScalingData/scale_test.data)



#### OCR

OCR is a practical example of Optical Character Recognition using FANN. While this example is limited and does make mistakes, the concepts illustrated by OCR can be applied to a more robust stacked network that uses feature extraction and convolution layers to recognize text of any font in any size image. 

[train_ocr.php](examples/ocr/train_ocr.php)

[test_ocr.php](examples/ocr/test_ocr.php)



#### Pathfinder

Pathfinder is an example of a neural network that is capable of plotting an 8 direction step path from a starting position in a 5x5 grid to an ending position in that grid. To keep the Pathfinder example simple it is not trained to deal with walls or non-traversable terrain however it would be very easy to add that by adding additional training.

[pathfinder_train.php](examples/pathfinder/pathfinder_train.php)

[pathfinder_test.php](examples/pathfinder/pathfinder_test.php)




