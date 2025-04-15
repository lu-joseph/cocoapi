from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

compile_args = ['-Wno-cpp', '-Wno-unused-function', '-std=c99'] if not sys.platform == 'win32' else []

ext_modules = cythonize([
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs=[np.get_include(), '../common'],
        extra_compile_args=compile_args,
    )
])

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir={'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0'
    ],
    version='2.0',
    ext_modules=ext_modules
)
