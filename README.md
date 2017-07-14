# The OrMachine

## Installation
Code is full compatible with Python 2 and 3.
To install and test the packge run the following commands:
```
>>> git clone https://github.com/TammoR/OrMachine
>>> cd OrMachine/
>>> pip3 install .
>>> python3 tests/test_ormachine.py
```
This test should finish about one second or less.

If you like to compile the Cython coude yourself
```
>>> python3 setup.py build_ext --inplace
```

Multi-core support comes from OpenMP. For single core use, remove the corresponding flags in the setup.py.

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

## Convergence issues
Should you experience lack of convergence (e.g. one factor matrix all 'off', the other uniform random),
try initialising the larger factor matrix mostly to ones and the smaller factor matrix to zeros. 
E.g. by defining the hidden layer as follows:
```
hidden = orm.add_layer(size=size, child=data, lbda_init=1.6, z_init=.9, u_init=.1)
```
You may also try to fix lbda for hold lbda fixed for the first few iteration:
```
hidden = orm.add_layer(size=size, child=data, lbda_init=1.5, z_init=.9, u_init=.1)
orm.infer(burn_in_max=500, fix_lbda_iters=50)
```


## Mac OS X troubleshooting
If you get an error ```clang: error: : errrorunsupported option '-fopenmp'```,
point to gcc as your default compiler (possibly after
installing it using homebrew: ```brew instal lgcc```), e.g. with
```
export CC=/usr/local/bin/g++-7
```


### OpenMP support with clang

Currently OpenMP support is not working. A promising route to resolving this is to build your own clang compiler which supports OpenMP, described [here](https://clang.llvm.org/get_started.html).

Choose your compiler by setting the ``CC`` variable, here
``export CC=/usr/local/myclang/build/bin/clang-5.0 ``

This still throws an error that it can't find ``omp.h``.
Setting ``C_INCLUDE_PATH`` to directories that include this file, e.g.

* ``/usr/local/Cellar/gcc/7.1.0/lib/gcc/7/gcc/x86_64-apple-darwin16.5.0/7.1.0/include``
* ``/usr/local/Cellar/llvm/4.0.0_1/lib/clang/4.0.0/include/``

results in 
`` error: __float128 is not supported on this target``.
This persists after removing the corresponding lines from ``c++config.h`` as suggested [on stackoverflow](https://stackoverflow.com/questions/43316533/float128-is-not-supported-on-this-target).

### OpenMP support with gcc
Set gcc as compiler with
``export CC=/usr/local/bin/gcc-7``.
This should work out of the box, but throws
``cc-7: error: unrecognized command line option '-Wshorten-64-to-32'``
independent of whether we use OpenMP or not. More about this error on [stackoverflow](https://github.com/rbenv/ruby-build/issues/325)

Seems to be solved [here](https://stackoverflow.com/questions/40234807/python-c-extension-with-openmp-for-os-x). Not sure how to apply this.


