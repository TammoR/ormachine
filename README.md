# The OrMachine

## Dependencies
Requires numpy, openmp
## Installation
To install and test the packge run the following commands:
```
>>> git clone https://github.com/TammoR/ormachine
>>> cd ormachine/
>>> pip3 install . --user
	or >>> python3 setup.py install --user
>>> python3 tests/test_ormachine.py
```
If you like to compile the Cython coude yourself
```
>>> python3 setup_cythonize.py build_ext --inplace
```

## Mac OS X troubleshooting
If you get an error ```clang: error: : errrorunsupported option '-fopenmp'```,
point to gcc as your default compiler (possile after installing it using homebrew), e.g. with
```
export CC=/usr/local/bin/g++-7
```

FOr further MAC troubleshooting see mac_compilation.md.


## Basic usage example for a single layer Boolean Matrix Factorisation
See examples folder for jupyter notebooks.

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
