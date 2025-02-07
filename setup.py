from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import shutil

extensions = [
    Extension(
        "aimbot",
        ["aim_alignment.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="aimbot",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True,
        }
    ),
)
# BUILD CMD: python setup.py build_ext --inplace