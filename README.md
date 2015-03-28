HeteroSampler
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



Dependencies (OS X, Homebrew)
-----------------------------

Installation
-------------
```
cmake .
sudo make 
```


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






