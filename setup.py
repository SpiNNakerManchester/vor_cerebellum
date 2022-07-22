# Copyright (c) 2019-2022 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'matplotlib',
    'spinngym',
    'brian2',
    'quantities==0.12.4',
    'elephant==0.7.0',
    'neo==0.8.0',
    'argparse',
]

setup(
    name='vor_cerebellum',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/SpiNNakerManchester/vor_cerebellum',
    license="GNU GPLv3.0",
    author='Petrut Antoniu Bogdan',
    author_email='petrut.bogdan@manchester.ac.uk',
    description='Simulating a small scale Cerebellum model on '
                'SpiNNaker with online learning to perform the '
                'vestibulo-ocular reflex',
    # Requirements
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",

        "Intended Audience :: Science/Research",

        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 3"
        "Programming Language :: Python :: 3.7"

        "Topic :: Scientific/Engineering",
    ]
)
