# The OrMachine

## Dependencies
Requires python3-devel, Cython, numpy and scipy.

## Installation
To install and test the packge run the following commands:
```
>>> git clone https://github.com/TammoR/ormachine
>>> cd ormachine/
>>> pip3 install .
	or >>> python3 setup.py install
	(if you don't have the right permission add --user)
>>> python3 tests/test_ormachine.py
```

## Basic usage example for a single layer Boolean Matrix Factorisation
```
import ormachine
import numpy as np

# generate toy data in {-1,1} domain
X = 2*np.array([[0,0,1,1,0,0],[1,1,0,0,0,0],[0,0,1,1,1,1]])-1
X = np.concatenate(200*[X])

# invoke machine object
orm = ormachine.machine()
data = orm.add_matrix(val=X, sampling_indicator=False)

# add layer 
layer1 = orm.add_layer(size=3, child=data, lbda_init=2)

# run inference
orm.infer()
```
