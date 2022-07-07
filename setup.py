# Copyright (C) Cyprien Hoelzl
#
# This file is part of railFE.
#
# railFE is free software: you can redistribute it and/or modify
# it under the terms of the MIT License.
#
# railFE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the MIT License along with railFE. 

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

try:
    import numpy as np
except ImportError:
    raise ImportError('railFE requires the NumPy package.')

import glob
import os
import pathlib
import re
import subprocess
import shutil
import sys

# Importing _version__.py before building can cause issues.
with open('railFE/_version.py', 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)

# Parse package name from init file. Importing __init__.py / railFE will break as railFE depends on compiled .pyx files.
with open('railFE/__init__.py', 'r') as fd:
    packagename = re.search(r'^__name__\s*=\s*[\'"]([^\'"]*)[\'"]',
                            fd.read(), re.MULTILINE).group(1)


packages = [packagename, 'examples']

# Parse long_description from README.rst file.
with open('README.MD','r') as fd:
    long_description = fd.read()

# Python version
if sys.version_info[:2] < (3, 4):
    sys.exit('\nExited: Requires Python 3.4 or newer!\n')

extensions = []

 # SetupTools Required to make package
import setuptools

setup(name=packagename,
      version=version,
      author='Cyprien Hoelzl',
      url='https://github.com/CyprienHoelzl/railFE',
      description='Dynamic simulation of a simplified rail vehicle rolling on a railway track',
      long_description=long_description,
      long_description_content_type="text/x-rst",
      license='MIT',
      classifiers=[
          'Environment :: Console',
          'License :: OSI Approved :: MIT',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering'
      ],
      #requirements
      python_requires=">3.6",
      install_requires=[
          "h5py",
          "matplotlib",
          "numpy",
          "scipy",
          "control"
          ],
      ext_modules=extensions,
      packages=packages,
      include_package_data=True,
      include_dirs=[np.get_include()],
      zip_safe=False)