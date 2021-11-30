from setuptools import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

extensions = [
    Extension('triangle_hash', ['*.pyx'], include_dirs = [numpy.get_include()]),
    ]

setup(
    ext_modules = cythonize(extensions)
    )