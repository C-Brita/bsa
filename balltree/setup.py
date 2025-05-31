from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import platform

extra_compile_args = ['-O3', '-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin': # macOS
    brew_prefix = os.popen('brew --prefix').read().strip()
    libomp_path = f"{brew_prefix}/opt/libomp"
    
    extra_compile_args += ['-Xpreprocessor', '-fopenmp', f'-I{libomp_path}/include']
    extra_link_args += [f'-L{libomp_path}/lib', '-lomp']
else:
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']

ext_modules = [
    Extension(
        "balltree",
        ["balltree.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++"
    )
]

setup(
    name="balltree",
    ext_modules=cythonize(ext_modules),
    install_requires=['numpy']
)