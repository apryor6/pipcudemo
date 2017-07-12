from setuptools import setup, Extension

pipcudemo_core = Extension('pipcudemo.core',
	sources=['pipcudemo/core.cpp'],
	include_dirs = ['.'],
	libraries=['mylib'])

setup(name = 'pipcudemo',
	author = 'Alan (AJ) Pryor, Jr.', 
	author_email='apryor6@gmail.com',
	version = '1.0.3',
	description="An example project showing how to build a pip-installable Python package that invokes custom CUDA/C++ code.",
	ext_modules=[pipcudemo_core],
	packages=['pipcudemo'])
