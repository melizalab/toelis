#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
if sys.hexversion < 0x02060000:
    raise RuntimeError("Python 2.6 or higher required")

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup

# --- Distutils setup and metadata --------------------------------------------

VERSION = '2.0.0'

cls_txt = \
"""
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

short_desc = "Time of Event data and file format"

long_desc = """
Library for I/O and processing of time of event data. The toelis data structure
and file format are designed for storing the times of neural spikes emitted in
response to presented stimuli. Files can store multiple repeats from multiple
units.
"""

setup(
    name = 'toelis',
    version = VERSION,
    description = short_desc,
    long_description = long_desc,
    classifiers = [x for x in cls_txt.split("\n") if x],
    author = 'Dan Meliza',
    author_email = '"dan" at the domain "meliza.org"',
    maintainer = 'Dan Meliza',
    maintainer_email = '"dan" at the domain "meliza.org"',
    url = "https://github.com/melizalab/toelis",
    download_url="https://github.com/melizalab/toelis/downloads",

    py_modules = ['toelis'],
    test_suite='nose.collector'
    )
# Variables:
# End:
