HeteroSampler [![Build Status](https://travis-ci.org/strin/HeteroSampler.svg?branch=release)](https://travis-ci.org/strin/HeteroSampler)
=============

This project is a C++ implementation of HeteroSamplers: heterogenous Gibbs samplers for structured prediction problems. It is based on algorithms published in the AISTATS 2015 paper "Learning Where to Sample in Structured Prediction" by Shi Tianlin, Jacob Steinhardt, and Percy Liang.

How does it work
----------------
Taking a pre-trained model and its Gibbs sampler, the algorithm uses reinforcement learning to figure out which part of the structured output needs more sampling, and hence require more computational resources.


Installing HeteroSampler
-------------
This release is for early adopters of this premature software. Please let us know if you have comments or suggestions. Contact: tianlinshi [AT] gmail.com


HeteroSampler is written in C++ 11, so requires gcc >= 4.8. It also uses HDF5 for reading some type of model data. It is partially built on <a href="http://hci.iwr.uni-heidelberg.de/opengm2/">OpenGM</a>, a open-source graphical model toolbox.

Dependencies (Ubuntu)
---------------------
To install gcc 4.8,

```
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test;
sudo apt-get update -qq
sudo apt-get install -qq g++-4.8
export CXX="g++-4.8"
```

Install cmake to bulid the source code

```
sudo apt-get install cmake
```

Install boost-program-options

```
sudo add-apt-repository -y ppa:boost-latest/ppa
sudo apt-get update -qq
sudo apt-get install libboost-all-dev
sudo apt-get install libboost1.55-all-dev
```

Install Hierarchical Data Format (HDF 5):

```
sudo apt-get install libhdf5-serial-dev
```


Dependencies (OS X, Homebrew)
-----------------------------

Installation
-------------
```
cmake .
make
```

Example
=======

Pre-Train a Sequence Tagging Model
-----------------------------------------------------

To pre-train an NER model, run <i>tagging</i> with the following parameters:

```
./tagging --T 8 --B 5 --train data/eng_ner/train --test data/eng_ner/test --eta 0.3 --depthL 2 --windowL 2 --factorL 2 --output model/eng_ner/gibbs.model --scoring NER --Q 1 --log 'log/eng_ner/log' --testFrequency 1
```

| Parameters | Meaning |
|------------------|--------------|
|        T            |   number of Gibbs sweeps over each training instance |
|        B            |   number of burn-in steps for Gibbs sweeps |


Citation
========
If our software helps you in your work, please cite

<i>Shi, Tianlin, Jacob Steinhardt, and Percy Liang. "Learning Where to Sample in Structured Prediction." Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics. 2015.</i>

License (GPL V3)
========

Copyright (C) 2014 Tianlin Shi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
