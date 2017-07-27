from setuptools import setup, Extension
from Cython.Build import cythonize
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

def _is_gcc(compiler):
    return "gcc" in compiler or "g++" in compiler

class build_ext_subclass( build_ext ):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c == "unix":
            compiler_args = self.compiler.compiler
            c = compiler_args[0]  # get the compiler name (argv0)
            if _is_gcc(c):
                names = [c, "gcc"]
                # Fix up a problem on older Mac machines where Python
                # was compiled with clang-specific options:
                #  error: unrecognized command line option '-Wshorten-64-to-32'
                compiler_so_args = self.compiler.compiler_so
                for args in (compiler_args, compiler_so_args):
                    if "-Wshorten-64-to-32" in args:
                        del args[args.index("-Wshorten-64-to-32")]

        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        "cython_functions",
        ["cython_functions.pyx"],
        extra_compile_args = ["-Ofast", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name='ormachine',
    cmdclass = {"build_ext": build_ext_subclass},
    version='0.1',
    author='Tammo Rukat',
    author_email='tammorukat@gmail.com',
    url='https://github.com/TammoR/OrMachine',
    py_modules=['ormachine','cython_functions','wrappers'],
    ext_modules=cythonize(ext_modules),
    setup_requires=["Cython >= 0.20"],
    install_requires=['numpy','scipy','Cython']
)



