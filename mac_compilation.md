# Compiling OrMachine on MAC

[OrMachine on gitHub](https://github.com/TammoR/OrMachine/)

Very similar issue on [stackoverflow](https://stackoverflow.com/questions/40234807/python-c-extension-with-openmp-for-os-x)

## OpenMP support with clang

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

## OpenMP support with gcc
Set gcc as compiler with
``export CC=/usr/local/bin/gcc-7``.
This should work out of the box, but throws
``cc-7: error: unrecognized command line option '-Wshorten-64-to-32'``
independent of whether we use OpenMP or not. More about this error on [stackoverflow](https://github.com/rbenv/ruby-build/issues/325)

Seems to be solved [here](https://stackoverflow.com/questions/40234807/python-c-extension-with-openmp-for-os-x). Not sure how to apply this.

## Misc
* Do we cast types when reading out array shape (64 to 32 bit or sth?)