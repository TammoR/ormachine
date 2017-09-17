from Cython.Build import cythonize
from setuptools import setup, Extension
#from distutils.extension import Extension
#from Cython.Distutils import build_ext
import numpy



ext_modules = [
    Extension(
        "cython_functions", ["cython_functions.pyx"],
	include_dirs=[numpy.get_include()],
    extra_compile_args = ["-Ofast", "-ffast-math", "-march=native", "-fopenmp"],
	extra_link_args=['-fopenmp']
    )
]

setup(
    name='ormachine',
    version='0.1',
    author='Tammo Rukat',
    author_email='tammorukat@gmail.com',
    url='https://github.com/TammoR/OrMachine',
    py_modules=['ormachine','wrappers'],
    # install_requires=['numpy','Cython'],
    ext_modules=cythonize(ext_modules)
)



