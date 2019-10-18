from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

# # define an extension that will be cythonized and compiled
# ext = Extension(name="hamiltonian", sources=["hamiltonian.pyx"])
# setup(ext_modules=cythonize(ext))

setup(
    ext_modules=cythonize("hamiltonian.pyx"),
    include_dirs=[numpy.get_include()]
)  