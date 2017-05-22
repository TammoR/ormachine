from setuptools import setup, Extension
from Cython.Build import cythonize
from distutils.core import setup
#from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension(
        "cython_functions", ["cython_functions.c"],
	include_dirs=[numpy.get_include()]
    )
]

setup(
    name='ormachine',
    version='0.1',
    author='Tammo Rukat',
    author_email='tammorukat@gmail.com',
    url='https://github.com/TammoR/OrMachine',
    py_modules=['ormachine'],
    ext_modules=cythonize(ext_modules)
)



